"""
DTW stands for dynamic time warping distance, which is a
measurement of how far two time series are.
"""

import numpy as np

from sklearn.metrics.pairwise import check_pairwise_arrays

from ...utils._check import check_valid_option

_ALLOWED_MEHTODS = ['LB_Kim', 'LB_Keogh', 'LB_Keogh_Reversed']


def _DTW_wrapper(X, Y, m, n, history, weighted):
    r"""Actual calculation wrapper for DTW. The wrapper makes
    the calculation in a dynamic programming manner.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples, n_features)
        The second input data.
    m : int
        The first input data index.
    n : int
        The second input data index.
    history : dict
        The stored calculation of step `m` and `n`.
    weighted : bool
        If weighted, the diagnoal factor is 2, if not, the factor is 1.

    Returns
    -------
    distance : float
        The distance between X and Y.
    """

    if (m, n) in history:
        return history[(m, n)]

    if weighted:
        factor = 2
    else:
        factor = 1

    w = np.linalg.norm(X[m] - Y[n], ord=2)

    if m > 0 and n > 0:
        result = min(
            w + _DTW_wrapper(X, Y, m - 1, n, history, weighted),
            w + _DTW_wrapper(X, Y, m, n - 1, history, weighted),
            factor * w + _DTW_wrapper(
                X, Y, m - 1, n - 1, history, weighted
            )
        )
    elif m == 0 and n > 0:
        result = w + _DTW_wrapper(X, Y, m, n - 1, history, weighted)
    elif n == 0 and m > 0:
        result = w + _DTW_wrapper(X, Y, m - 1, n, history, weighted)
    else:
        result = w + 0

    if (m, n) not in history:
        history[(m, n)] = result

    return result


def dtw(X, Y=None, weighted=True):
    r"""In time series analysis, dynamic time warping (DTW) is one of
    the algorithms for measuring similarity between two temporal sequences,
    which may vary in speed. In general, DTW is a method that calculates an
    optimal match between two given sequences (e.g. time series) with
    certain restriction and rules.

    For two time series,

        .. math::
            X = [x_1, x_2, x_3, \ldots, x_m] \\
            Y = [y_1, y_2, y_3, \ldots, y_n]

    we calculate the point-wise distance x_{i} and y_{j} as,

        .. math::
            w(i, j) = (x_{i}, y_{j})^{2}

    so for any warping path, notes as W, we define the k-th element
    of W as,

        .. math::
            W_{k} = (i, j)_{k}

    so we warping path W,

        .. math::
            W = w_{1}, w_{2}, \ldots, w_{k}, \max \{ m, n \} \leq K \lt m+n-1

    we calculate distance D(X, Y) as,

        .. math::
            D(X, Y) = D_{m, n} = W(m, n) + \min \{ D_{m-1,n},
            D_{m,n-1}, D_{m-1,n-1} \}

    as for the weighted version,

        .. math::
            D(X, Y) = D_{m, n} = W(m, n) + \min \{ D_{m-1,n},
            D_{m,n-1}, 2D_{m-1,n-1} \}

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        The second input data.
    weighted : bool, optional
        Use weighted dtw version or not.

    Returns
    -------
    distance : float
        The distance between X and Y.
    """

    X, Y = check_pairwise_arrays(X, Y)

    m, n = len(X) - 1, len(Y) - 1
    history = {}

    distance = _DTW_wrapper(
        X, Y, m, n, history,
        weighted=weighted
    )

    return distance


def _lb_keogh_estimate(X, Y, r=1):
    r"""The LB_Keogh method to estimate the lower bound of DTW.

    For LB_Keogh, we first define a `reach` called, `r`, which serves as a
    constraint, allowed range of warping. For every point in X, we can
    calculate corresponding upper and lower bound as,

        .. math::
            u_i = \max x_{i-r}, \ldots, x_{i+r} \\
            l_i = \min x_{i-r}, \ldots, x_{i+r}

    using this point-wise upper and lower bound, we can calculate
    LB_Keogh as,

        .. math::
            LB\_Keogh(X, Y) = \sqrt{\sum_{i}^{n}\begin{cases}
                (y_i-u_i), & if \quad y_i \gt u_i \\
                (y_i-l_i), & if \quad y_i \lt u_i \\
                0, & otherwise
                \end{cases}}

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples, n_features)
        The second input data.
    r : int, optional
        The reach parameter served as a constraint.

    Returns
    -------
    lb : float
        The lower bound of the estimated DTW distance between X and Y.
    """

    n = min(len(X), len(Y))
    lb = 0

    for i in range(n):
        lower_range = 0 if i - r < 0 else i - r
        upper_range = n - 1 if i + r > n - 1 else i + r

        lower_bound = np.amin(X[lower_range:upper_range + 1], axis=0)
        upper_bound = np.amax(X[lower_range:upper_range + 1], axis=0)

        delta = []
        for j in range(len(Y[i])):
            if Y[i][j] < lower_bound[j]:
                delta.append(lower_bound[j] - Y[i][j])
            elif Y[i][j] > upper_bound[j]:
                delta.append(Y[i][j] - upper_bound[j])
            else:
                delta.append(0)

        lb += np.linalg.norm(delta, ord=2)

    return lb


def dtw_estimate(X, Y=None, method=None, r=1):
    r"""In time series analysis, dynamic time warping (DTW) is one of
    the algorithms for measuring similarity between two temporal sequences,
    which may vary in speed. However, DTW algorithm is rather time costing,
    its time and space complexity is O(nm). Therefore, we use lower bound
    estimation in order to accelerate DTW computation in large dataset.

    For two time series,

        .. math::
            X = [x_1, x_2, x_3, \ldots, x_n] \\
            Y = [y_1, y_2, y_3, \ldots, y_n]

    we estimate the lower bound of DTW with different methods, including,

        1. LB_Kim
        2. LB_Keogh
        3. LB_Keogh_Reversed

    For LB_Kim, we can retrieve two feature vectors of X and Y with,

        .. math::
            X\_feature = [x_0, x_n, \max X, \min X] \\
            Y\_feature = [y_0, y_n, \max Y, \min Y]

    the lower bound of LB_Kim is the maximum squared difference of these
    two feature vectors, as

        .. math::
            LB\_Kim(X, Y) = \max_{i} d(X\_feature_i, Y\_feature_i)

    For LB_Keogh, we first define a `reach` called, `r`, which serves as a
    constraint, allowed range of warping. For every point in X, we can
    calculate corresponding upper and lower bound as,

        .. math::
            u_i = \max x_{i-r}, \ldots, x_{i+r} \\
            l_i = \min x_{i-r}, \ldots, x_{i+r}

    using this point-wise upper and lower bound, we can calculate
    LB_Keogh as,

        .. math::
            LB\_Keogh(X, Y) = \sqrt{\sum_{i}^{n}\begin{cases}
                (y_i-u_i), & if \quad y_i \gt u_i \\
                (y_i-l_i), & if \quad y_i \lt u_i \\
                0, & otherwise
                \end{cases}}

    For LB_Keogh_Reversed, it is simply reversed version of LB_Keogh

        .. math::
            LB\_Keogh\_Reversed(X, Y) = LB\_Keogh(Y, X)

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        The second input data.
    method : str, optional
        The method used to estimate DTW distance.
        Currently supports,

            - LB_Kim
            - LB_Keogh
            - LB_Keogh_Reversed
    r : int, optional
        The reach parameter served as a constraint.

    Returns
    -------
    lb : float
        The lower bound of the estimated DTW distance between X and Y.

    Raises
    ------
    ValueError
        When method is not in `LB_Kim`, `LB_Keogh`, `LB_Keogh_Reversed`,
        a ValueError is raised.
    """

    X, Y = check_pairwise_arrays(X, Y)
    method = check_valid_option(
        method,
        options=_ALLOWED_MEHTODS,
        variable_name='DTW estimation method'
    )

    if method == 'LB_Kim':
        lb = max(
            np.linalg.norm(X[0] - Y[0], ord=2),
            np.linalg.norm(X[-1] - Y[-1], ord=2),
            np.linalg.norm(np.amax(X, axis=0) - np.amax(Y, axis=0), ord=2),
            np.linalg.norm(np.amin(X, axis=0) - np.amin(Y, axis=0), ord=2))
    elif method == 'LB_Keogh':
        lb = _lb_keogh_estimate(X, Y, r=r)
    elif method == 'LB_Keogh_Reversed':
        lb = _lb_keogh_estimate(Y, X, r=r)

    return lb
