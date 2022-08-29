"""
LP Norm is the most common distance calculation method
used to calculate the Euclidean distance at certain
dimension.
"""

import numpy as np

from sklearn.metrics.pairwise import check_pairwise_arrays


def lp_norm(X, Y=None, p=2):
    r"""LpNorm can calculate two vectors distance under Lp Space,
    for instance Euclidean distance is a special case of distance under
    L2-norm calculation. For general `p`, LpNorm is calculated as,

        .. math::
            \|X-Y\|_P = \frac{1}{n} \sum_{i=1}^{n} \|X_{i} - Y_{i}\|_P =
            \frac{1}{n} \sum_{i=1}^{n}
            (\sum_{j=1}^{k} |X_{i, j}-Y_{i,j}|^p)^{1/p}

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        The second input data.
    p : int, optional
        The p is used to specify the Lp space to use to calculate.

    Returns
    -------
    distance : float
        The distance between X and Y.
    """

    X, Y = check_pairwise_arrays(X, Y)

    distance = np.mean(
        np.linalg.norm(X - Y, ord=p, axis=1)
    )

    return distance
