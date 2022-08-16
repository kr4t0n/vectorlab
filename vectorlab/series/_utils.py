"""
Basic utility functions operated on series data.
"""

import numpy as np

from ..utils._check import check_1d_array


def estimate_next(X):
    r"""This is a naive method to estimate the next value in
    a series data.

    This method use the average gradients of previous data points
    to estimate the next point. For a series, :math:`x_1, x_2, \dots,
    x_n`, the next value :math:`x_{n+1}` is calculate by

        .. math::
            \bar{g} = \dfrac{1}{m} \sum\limits^m_{i=1} g(x_n, x_{n-i})\\
            x_{n+1} = x_{n-m+1} + \bar{g} \cdot m

    where :math:`g_{x_i, x_j}` denotes the gradient of the straight line
    between point :math:`x_i` and :math:`x_j`, and :math:`\bar{g}` represents
    the average points considered.

    Parameters
    ----------
    X : array_like, shape (n_samples)
        The input array.

    Returns
    -------
    float
        The next estimated value.

    Raises
    ------
    ValueError
        Since we need at least two values to estimate the next value,
        if the length of input array X is smaller than two,
        a ValueError is raised.
    """

    X = check_1d_array(X)

    if X.shape[0] <= 1:
        raise ValueError(
            'Length of series should be larger than 2 to estimate'
        )

    slopes = [
        (X[-1] - v) / (X.shape[0] - 1 - i)
        for i, v in enumerate(X[:-1])
    ]

    return X[1] + sum(slopes)


def extend_series(X,
                  extend_num=0, look_ahead=5):
    r"""This is a naive method to extend a series data. In this function,
    it will extend next value using estimate_next naive method in an iterative
    way.

    Parameters
    ----------
    X : array_like, shape (n_samples)
        The input array.
    extend_num : int, optional
        The number of extended points.
    look_ahead : int, optional
        The number of previous points to be considered.

    Returns
    -------
    X : array_like, shape (n_samples + extend_num)
        The extended array.

    Raises
    ------
    ValueError
        If extend_num is smaller than 0 and extend_num is not an integer,
        a ValueError is raised.
        If look_ahead is smaller than 2 and look_ahead is not an integer,
        a ValueError is raised.
    """

    if extend_num < 0 or not isinstance(extend_num, int):
        raise ValueError(
            'Length of extend_num should be an integer larger or equal'
            'to zero. Current extend_num is {}.'.format(extend_num)
        )

    if look_ahead < 2 or not isinstance(look_ahead, int):
        raise ValueError(
            'Length of look_head should be an integer larger or equal'
            'to two. Current look_head is {}.'.format(look_ahead)
        )

    X = check_1d_array(X)

    X_new = X.copy()

    for i in range(extend_num):
        extension = [estimate_next(X_new[-look_ahead:])]
        X_new = np.concatenate((X_new, extension))

    return X_new
