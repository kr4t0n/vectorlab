"""
Basic utility functions operated on series data.
"""

import numpy as np

from ..utils._check import check_nd_array, check_valid_int


def estimate_next(series):
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
    series : array_like, shape (n_samples)
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

    series = check_nd_array(series, n=1)

    if series.shape[0] <= 1:
        raise ValueError(
            'Length of series should be larger than 2 to estimate'
        )

    slopes = [
        (series[-1] - v) / (series.shape[0] - 1 - i)
        for i, v in enumerate(series[:-1])
    ]

    return series[1] + sum(slopes)


def extend_series(series,
                  extend_num=0, look_ahead=5):
    r"""This is a naive method to extend a series data. In this function,
    it will extend next value using estimate_next naive method in an iterative
    way.

    Parameters
    ----------
    series : array_like, shape (n_samples)
        The input array.
    extend_num : int, optional
        The number of extended points.
    look_ahead : int, optional
        The number of previous points to be considered.

    Returns
    -------
    series : array_like, shape (n_samples + extend_num)
        The extended array.

    Raises
    ------
    ValueError
        If extend_num is smaller than 0 and extend_num is not an integer,
        a ValueError is raised.
        If look_ahead is smaller than 2 and look_ahead is not an integer,
        a ValueError is raised.
    """

    series = check_nd_array(series, n=1)

    extend_num = check_valid_int(
        extend_num,
        lower=0, variable_name='extend_num'
    )
    look_ahead = check_valid_int(
        look_ahead,
        lower=2, variable_name='look_ahead'
    )

    series_new = series.copy()

    for i in range(extend_num):
        extension = [estimate_next(series_new[-look_ahead:])]
        series_new = np.concatenate((series_new, extension))

    return series_new
