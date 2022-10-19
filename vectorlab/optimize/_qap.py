"""
This module provides a solution using first order optimization
method to solve lowler form of quadratic assignment problem.
"""

import operator
import numpy as np

from scipy._lib._util import check_random_state
from scipy.optimize._optimize import _check_unknown_options
from scipy.optimize._qap import _common_input_validation
from scipy.optimize._qap import _doubly_stochastic, _calc_score
from scipy.optimize import linear_sum_assignment, OptimizeResult


def _quadratic_assignment_lawler(A, B, C=None,
                                 maximize=False, partial_match=None, rng=None,
                                 P0='barycenter', shuffle_input=False,
                                 maxiter=30, tol=0.03,
                                 aggr='sum',
                                 **unknown_options):
    r"""Solve the quadratic assignment problem (approximately).

    Original paper is https://ieeexplore.ieee.org/document/6909665.

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the first order optimization method.

    Quadratic Assignment solves problems of the following form:

    .. math::
        \min_P \text{trace}(P^T A P B) \\
        \text{s.t.} \ P \in \mathcal{P} \\

    However, such form can also be expressed in a more general and compact
    form

    .. math::
        \min_P vec(P)^T C vec(P) \\
        \text{s.t.} \ P \in \mathcal{P} \\
        \text{where} \ C = B \otimes A \\

    Graph matching tries to *maximize* the objective function. The algorithm
    can be thought of as finding the alignment of the nodes of two graphs
    that minimizes the number of induced edge disagreement, or, in the case
    of weighted graphs, the sum of squared edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximates and are not guaranteed to be optimal.

    Parameters
    ----------
    A : array_like, shape (n_nodes, n_nodes)
        The square matrix :math:`A` in the objective function above.
    B : array_like, shape (n_nodes, n_nodes)
        The square matrix :math:`B` in the objective function above.
    C : array_like, shape (n_nodes * n_nodes, n_nodes * n_nodes), optional
        The square matrix :math:`C` in the objective function above.
        If can be pre-computed or it will use the Kronecker result of
        matrix :math:`A` and :math:`B`.
    maximize : bool, optional
        Maximize the objective function if `True`.
    partial_match : array_like, shape (m_nodes, 2), optional
        Fixes part of the matching. Also known as a `seed`.

        Each row of `partial_match` specifies a pair of matched nodes:
        node ``partial_match[i, 0]`` of `A` is matched to node
        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``, where
        ``m`` is not greater than the number of nodes, :math:`n`.
    rng : None, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    P0 : array_like, shape (n-m, n-m), or str, optional
        Initial position. Must be a doubly-stochastic matrix.

        If the initial position is an array, it must be a doubly stochastic
        matrix of size :math:`m' \times m'` where :math:`m' = n - m`.

        If ``"barycenter"`` (default), the initial position is the barycenter
        of the Birkhoff polytope (the space of doubly stochastic matrices).
        This is a :math:`m' \times m'` matrix with all entries equal to
        :math:`1 / m'`.

        If ``"randomized"`` the initial search position is
        :math:`P_0 = (J + K) / 2`, where :math:`J` is the barycenter and
        :math:`K` is a random doubly stochastic matrix.
    shuffle_input : bool, optional
        Set to `True` to resolve degenerate gradients randomly. For
        non-degenerate gradients this option has no effect.
    maxiter : int, optional
        Integer specifying the max number of iterations performed.
    tol : float, optional
        Tolerance for termination. Iteration terminates when
        :math:`\frac{||P_{i}-P_{i+1}||_F}{\sqrt{n - m}} \leq tol`,
        where :math:`i` is the iteration number.
    aggr : str, optional
        The aggregation method used to compute the matmul.
        Since the grad of update is equal to :math:`Cx`, aggregation
        method will control the way of such matmul operation result.

    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.

        col_ind : array_like, (n_nodes)
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of iterations performed.
    """

    _check_unknown_options(unknown_options)

    maxiter = operator.index(maxiter)

    # ValueError check
    # would support partial match in future
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    if C is None:
        C = np.kron(A, B)

    msg = None
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = 'Invalid `P0` parameter string'
    elif maxiter <= 0:
        msg = '`maxiter` must be a positive integer'
    elif tol <= 0:
        msg = '`tol` must a positive float'
    if msg is not None:
        raise ValueError(msg)

    rng = check_random_state(rng)
    n = len(A)
    n_seeds = len(partial_match)  # number of seeds
    n_unseed = n - n_seeds

    # choose initialization
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        if P0.shape != (n_unseed, n_unseed):
            msg = '`P0` matrix must have shape n x n'
        elif ((P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1)
              or not np.allclose(np.sum(P0, axis=1), 1)):
            msg = '`P0` matrix must be doubly stochastic'
        if msg is not None:
            raise ValueError(msg)
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        # generate a nxn matrix where each entry is a random number [0, 1]
        # would use rand, but Generators don't have it
        # would use random, but old mtrand.RandomStates don't have it
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        P0 = (J + K) / 2

    # check trivial cases
    if n == 0 or n_seeds == n:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {'col_ind': partial_match[:, 1], 'fun': score, 'nit': 0}
        return OptimizeResult(res)

    # if maximize, we should follow the gradient direction
    # otherwise, in the opposite direction
    obj_func_scalar = -1
    if maximize:
        obj_func_scalar = 1

    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)

    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    p = P0.flatten()
    # loop while stopping criteria not met
    for n_iter in range(1, maxiter + 1):

        # update p with c @ p / norm(c @ p)
        # if aggr method is sum, cp is simply as c @ p
        # if aggr method is max, cp will equal to diag + max(non diag),
        # since sum is, in fact, equal to diag + sum(non diag)
        if aggr == 'sum':
            cp = C @ p
        elif aggr == 'max':
            cp = C * p
            cp = cp.diagonal() + \
                np.apply_along_axis(
                    lambda x: x.reshape(
                        -1,
                        cp.shape[0]
                    ).max(axis=1).ravel().sum(),
                axis=1,
                arr=cp
            )

        p_i1 = obj_func_scalar * cp / np.linalg.norm(cp)

        if np.linalg.norm(p - p_i1) / np.sqrt(n_unseed) < tol:
            p = p_i1
            break

        p = p_i1
    # end main loop

    # reconstruct P matrix with vector p
    P = p.reshape(n, n)

    # project onto the set of permutation matrices
    _, cols = linear_sum_assignment(P, maximize=True)

    score = _calc_score(A, B, cols)
    res = {'col_ind': cols, 'fun': score, 'nit': n_iter}
    return OptimizeResult(res)


def _quadratic_assignment_yaaq(A, B, C=None,
                               maximize=False, partial_match=None, rng=None,
                               P0='barycenter', shuffle_input=False,
                               maxiter=30, tol=0.03,
                               **unknown_options):
    r"""Solve the quadratic assignment problem (approximately).

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the first order optimization method.

    Quadratic Assignment solves problems of the following form:

    .. math::
        \min_P \text{trace}(P^T A P B) \\
        \text{s.t.} \ P \in \mathcal{P} \\

    However, such form can also be expressed in a more general and compact
    form

    .. math::
        \min_P vec(P)^T C vec(P) \\
        \text{s.t.} \ P \in \mathcal{P} \\
        \text{where} \ C = B \otimes A \\

    Graph matching tries to *maximize* the objective function. The algorithm
    can be thought of as finding the alignment of the nodes of two graphs
    that minimizes the number of induced edge disagreement, or, in the case
    of weighted graphs, the sum of squared edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximates and are not guaranteed to be optimal.

    Parameters
    ----------
    A : array_like, shape (n_nodes, n_nodes)
        The square matrix :math:`A` in the objective function above.
    B : array_like, shape (n_nodes, n_nodes)
        The square matrix :math:`B` in the objective function above.
    C : array_like, shape (n_nodes * n_nodes, n_nodes * n_nodes), optional
        The square matrix :math:`C` in the objective function above.
        If can be pre-computed or it will use the Kronecker result of
        matrix :math:`A` and :math:`B`.
    maximize : bool, optional
        Maximize the objective function if `True`.
    partial_match : array_like, shape (m_nodes, 2), optional
        Fixes part of the matching. Also known as a `seed`.

        Each row of `partial_match` specifies a pair of matched nodes:
        node ``partial_match[i, 0]`` of `A` is matched to node
        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``, where
        ``m`` is not greater than the number of nodes, :math:`n`.
    rng : None, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    P0 : array_like, shape (n-m, n-m), or str, optional
        Initial position. Must be a doubly-stochastic matrix.

        If the initial position is an array, it must be a doubly stochastic
        matrix of size :math:`m' \times m'` where :math:`m' = n - m`.

        If ``"barycenter"`` (default), the initial position is the barycenter
        of the Birkhoff polytope (the space of doubly stochastic matrices).
        This is a :math:`m' \times m'` matrix with all entries equal to
        :math:`1 / m'`.

        If ``"randomized"`` the initial search position is
        :math:`P_0 = (J + K) / 2`, where :math:`J` is the barycenter and
        :math:`K` is a random doubly stochastic matrix.
    shuffle_input : bool, optional
        Set to `True` to resolve degenerate gradients randomly. For
        non-degenerate gradients this option has no effect.
    maxiter : int, optional
        Integer specifying the max number of iterations performed.
    tol : float, optional
        Tolerance for termination. Iteration terminates when
        :math:`\frac{||P_{i}-P_{i+1}||_F}{\sqrt{n - m}} \leq tol`,
        where :math:`i` is the iteration number.

    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.

        col_ind : array_like, (n_nodes)
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of iterations performed.
    """

    _check_unknown_options(unknown_options)

    maxiter = operator.index(maxiter)

    # ValueError check
    # would support partial match in future
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    if C is None:
        C = np.kron(A, B)

    msg = None
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = 'Invalid `P0` parameter string'
    elif maxiter <= 0:
        msg = '`maxiter` must be a positive integer'
    elif tol <= 0:
        msg = '`tol` must a positive float'
    if msg is not None:
        raise ValueError(msg)

    rng = check_random_state(rng)
    n = len(A)
    n_seeds = len(partial_match)  # number of seeds
    n_unseed = n - n_seeds

    # choose initialization
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        if P0.shape != (n_unseed, n_unseed):
            msg = '`P0` matrix must have shape n x n'
        elif ((P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1)
              or not np.allclose(np.sum(P0, axis=1), 1)):
            msg = '`P0` matrix must be doubly stochastic'
        if msg is not None:
            raise ValueError(msg)
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        # generate a nxn matrix where each entry is a random number [0, 1]
        # would use rand, but Generators don't have it
        # would use random, but old mtrand.RandomStates don't have it
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        P0 = (J + K) / 2

    # check trivial cases
    if n == 0 or n_seeds == n:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {'col_ind': partial_match[:, 1], 'fun': score, 'nit': 0}
        return OptimizeResult(res)

    # if maximize, we should follow the gradient direction
    # otherwise, in the opposite direction
    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1

    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)

    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    p = P0.flatten()
    # loop while stopping criteria not met
    for n_iter in range(1, maxiter + 1):

        # update p with c @ p / norm(c @ p)
        # if aggr method is sum, cp is simply as c @ p
        # if aggr method is max, cp will equal to diag + max(non diag),
        # since sum is, in fact, equal to diag + sum(non diag)
        grad_fp = C @ p + C.T @ p
        _, cols = linear_sum_assignment(
            grad_fp.reshape(n_unseed, n_unseed),
            maximize=maximize
        )
        q = np.eye(n_unseed)[cols].flatten()

        # Compute the first order Tylor Expansion
        a = (p - q).T @ C @ (p - q)
        b = q.T @ C @ (p - q) + (p - q).T @ C @ q

        if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * obj_func_scalar])

        p_i1 = alpha * p + (1 - alpha) * q
        if np.linalg.norm(p - p_i1) / np.sqrt(n_unseed) < tol:
            p = p_i1
            break

        p = p_i1
    # end main loop

    # reconstruct P matrix with vector p
    P = p.reshape(n, n)

    # project onto the set of permutation matrices
    _, cols = linear_sum_assignment(P, maximize=True)

    score = _calc_score(A, B, cols)
    res = {'col_ind': cols, 'fun': score, 'nit': n_iter}
    return OptimizeResult(res)
