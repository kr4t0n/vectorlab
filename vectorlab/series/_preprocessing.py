"""
Preprocessing functions operated on series data.
"""

import math
import numpy as np

from functools import reduce
from scipy import interpolate

from ..utils._check import (
    check_nd_array,
    check_valid_int, check_valid_float,
    check_valid_option
)

EPS = 1e-8


def auto_ts_step(ts, eps=4):
    r"""This function is aimed to choose the time step used in
    format_ts function. This function uses greatest common divisor
    (GCD) to find the most likely time step used to format the time
    stamp, aiming to expand least time stamps to make the time stamps
    be an arithmetic sequence.

    Parameters
    ----------
    ts : array_like, shape (n_samples)
        The input time stamps.
    eps : int, optional
        The accuracy to preserve time stamps if time stamps are float
        numbers.

    Returns
    -------
    ts_step : int, float
        The time step calculated by GCD.
    """

    ts = check_nd_array(ts, n=1)
    eps = check_valid_int(
        eps,
        lower=0, variable_name='eps'
    )

    sorted_index = np.argsort(ts)
    sorted_ts = ts[sorted_index]

    sorted_ts = (sorted_ts * (10 ** eps)).astype(np.int_)
    sorted_ts = sorted_ts - sorted_ts[0]

    ts_step = reduce(math.gcd, sorted_ts) / float(10 ** eps)

    return ts_step


def format_ts(ts, series, step, start_ts=None):
    r"""This function is aimed to format time stamp, and format the
    time series data at the same time. According to provided start_ts
    and desired the step, the new time stamps will be formatted as
    `start_ts, start_ts + step, start_ts + step * 2, ...`. New time
    series will be formatted according to the new order of time stamps.
    When certain time stamp data cannot be found, a `np.nan` will be
    used.

    Parameters
    ----------
    ts : array_like, shape (n_samples)
        The input time stamps.
    series : array_like, shape (n_featues, n_samples)
        The input time series data.
    step : int, float
        The step between two output time stamps.
    start_ts : int, float, optional
        The start time stamp.

    Returns
    -------
    new_ts : array_like, shape (n_samples)
        The output formatted time stamps.
    new_series : array_like, shape (n_featues, n_samples)
        The output formatted time series data.

    Raises
    ------
    ValueError
        When start time stamp is larger than any existed time stamps,
        a ValueError is raised.
    """

    ts = check_nd_array(ts, n=1)
    series = check_nd_array(series, n=2)
    step = check_valid_float(
        float(step),
        lower=0, variable_name='step'
    )

    sorted_index = np.argsort(ts)
    sorted_ts = ts[sorted_index]
    sorted_series = series[:, sorted_index]

    if start_ts is None:
        start_ts = sorted_ts[0]

    if start_ts > sorted_ts[-1]:
        raise ValueError(
            f'Start time stamp is {start_ts}, which is larger than any '
            f'time stamps.'
        )

    new_ts = np.arange(
        start_ts,
        sorted_ts[-1] + EPS,
        step,
        dtype=sorted_ts.dtype
    )
    new_series = np.empty(
        (series.shape[0], new_ts.shape[0]),
        dtype=np.float_
    )
    new_series[:] = np.NaN

    new_series[:, np.isin(new_ts, sorted_ts)] = \
        sorted_series[:, np.isin(sorted_ts, new_ts)]

    return new_ts, new_series


def aggregate_ts(ts, series, step, agg_type, start_ts=None):
    r"""This function is aimed to aggregate time stamp, and aggregate the
    time series data with a specified aggregation method at the same time.
    According to provided start_ts and desired the step, the new time stamps
    will be aggregated as`start_ts, start_ts + step, start_ts + step * 2,
    ...`. New time series will be aggregated according to the new order of
    time stamps.

    Parameters
    ----------
    ts : array_like, shape (n_samples)
        The input time stamps.
    series : array_like, shape (n_featues, n_samples)
        The input time series data.
    step : int, float
        The step between two output time stamps.
    agg_type : str
        The aggregation method used to aggregate time series data.
    start_ts : int, float, optional
        The start time stamp.

    Returns
    -------
    new_ts : array_like, shape (n_samples)
        The output aggregated time stamps.
    new_series : array_like, shape (n_featues, n_samples)
        The output aggregated time series data.

    Raises
    ------
    ValueError
        When start time stamp is larger than any existed time stamps,
        a ValueError is raised.
    """

    _agg_type_options = ['sum', 'mean']

    ts = check_nd_array(ts, n=1)
    series = check_nd_array(series, n=2)
    step = check_valid_float(
        float(step),
        lower=0, variable_name='step'
    )
    agg_type = check_valid_option(
        agg_type,
        options=_agg_type_options, variable_name='agg_type'
    )

    sorted_index = np.argsort(ts)
    sorted_ts = ts[sorted_index]
    sorted_series = series[:, sorted_index]

    if start_ts is None:
        start_ts = sorted_ts[0]

    if start_ts > sorted_ts[-1]:
        raise ValueError(
            f'Start time stamp is {start_ts}, which is larger than any '
            f'time stamps.'
        )

    new_ts = np.arange(
        start_ts,
        sorted_ts[-1] + EPS,
        step,
        dtype=sorted_ts.dtype
    )
    new_series = np.empty(
        (series.shape[0], new_ts.shape[0]),
        dtype=np.float_
    )

    split = np.searchsorted(sorted_ts, new_ts, side='left')

    if agg_type == 'sum':
        new_series = np.hstack(
            np.array(
                list(
                    map(
                        lambda x: np.sum(x, axis=1, keepdims=True),
                        np.split(sorted_series, split, axis=1)[1:]
                    )
                )
            )
        )
    elif agg_type == 'mean':
        new_series = np.hstack(
            np.array(
                list(
                    map(
                        lambda x: np.mean(x, axis=1, keepdims=True),
                        np.split(sorted_series, split, axis=1)[1:]
                    )
                )
            )
        )

    return new_ts, new_series


def _series_interpolate(ts, series, kind='linear'):
    r"""This is an interpolation function to fill in the `np.nan` value
    with certain methods, which will ensure that every series value is
    not missing. Interpolation method can be chosen using parameter
    `kind`. Supporting `kind` includes,

    - linear
    - nearest
    - nearest-up
    - zero
    - slinear
    - quadratic
    - cubic
    - previous
    - next

    Parameters
    ----------
    ts : array_like, shape (n_samples)
        The input time stamps.
    series : array_like, shape (n_samples)
        The input time series data.
    kind : str, optional
        The method used in interpolation.

    Returns
    -------
    new_series : array_like, shape (n_samples)
        The output interpolated time series data.
    """

    nan_indices = np.isnan(series)

    f = interpolate.interp1d(
        ts[~nan_indices],
        series[~nan_indices],
        kind=kind,
        fill_value='extrapolate'
    )

    new_series = f(ts)

    return new_series


def series_interpolate(ts, series, kind='linear'):
    r"""This is an interpolation function to fill in the `np.nan` value
    with certain methods, which will ensure that every series value is
    not missing. Interpolation method can be chosen using parameter
    `kind`. Supporting `kind` includes,

    - linear
    - nearest
    - nearest-up
    - zero
    - slinear
    - quadratic
    - cubic
    - previous
    - next

    In this function, `numpy` will vectorize the helper function
    `_series_interpolate` to support two dimensional series data.

    Parameters
    ----------
    ts : array_like, shape (n_samples)
        The input time stamps.
    series : array_like, shape (n_featues, n_samples)
        The input time series data.
    kind : str, optional
        The method used in interpolation.

    Returns
    -------
    new_series : array_like, shape (n_featues, n_samples)
        The output interpolated time series data.
    """

    _interpolate_options = [
        'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', 'next'
    ]

    ts = check_nd_array(ts, n=1)
    series = check_nd_array(series, n=2)
    kind = check_valid_option(
        kind,
        options=_interpolate_options, variable_name='interpolate kind'
    )

    new_series = np.vectorize(
        _series_interpolate,
        signature='(n),(n)->(n)',
        excluded=['kind']
    )(ts, series, kind=kind)

    return new_series
