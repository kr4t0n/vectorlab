"""
Conversion between various formats of time, timestamp, datetime.
"""

import pytz
import warnings

from datetime import datetime

from ._check import check_valid_float, check_valid_option


def _auto_convert_ts(ts):
    r"""Auto convert timestamp in to seconds

    Parameters
    ----------
    ts : float
        Input timestamp

    Returns
    -------
    ts : float
        Timestamp in seconds.
    """

    warnings.warn(
        'This is an experimental function to automatically '
        'convert your timestamp into seconds. It can be '
        'undesirable, and we encourage you to do it manually.'
    )

    ts_l = len(str(ts))

    if ts_l == 9 or ts_l == 10:
        return ts
    elif ts_l == 13:
        return ts / (10 ** 3)
    elif ts_l == 16:
        return ts / (10 ** 6)

    warnings.warn(
        f'Your timestamp may be invalid. Please check your time '
        f'stamp {ts}, and ensure it is based on seconds.'
    )

    return ts


def _convert_ts(ts, unit):
    r"""Convert timestamp in to seconds

    Parameters
    ----------
    ts : float
        Input timestamp

    Returns
    -------
    ts : float
        Timestamp in seconds.
    """

    if unit == 's':
        return ts
    elif unit == 'ms':
        return ts / (10 ** 3)
    elif unit == 'us':
        return ts / (10 ** 6)
    elif unit == 'auto':
        return _auto_convert_ts(ts)

    return ts


def dt_to_dttz(date, tz='HongKong'):
    r"""Convert a standard datetime object
    to a datetime object with specified time zone.

    Parameters
    ----------
    dt : datetime object
        Datetime object without a particular time zone.
    tz : str, optional
        Time zone specified to format a datetime object.

    Returns
    -------
    dttz : datetime object
        Datetime object with a particular time zone.
    """

    dttz = pytz.timezone(tz).localize(date)

    return dttz


def ts_to_dttz(ts, tz='HongKong', unit='s'):
    """Convert a timestamp to a datetime object with
    specified time zone.

    Parameters
    ----------
    ts : float
        The timestamp.
    tz : str, optional
        Specified time zone.
    unit : str, optional
        The unit of timestamp, it should be either `s`,
        `ms`, `us` or `auto`.

    Returns
    -------
    dttz : datetime object
        Datetime object with a particular time zone.
    """

    ts = check_valid_float(
        float(ts),
        variable_name='timestamp'
    )
    unit = check_valid_option(
        unit,
        options=['s', 'ms', 'us', 'auto'],
        variable_name='unit'
    )

    ts = _convert_ts(ts, unit)

    utc_dttz = pytz.timezone('UTC').localize(
        datetime.utcfromtimestamp(ts)
    )
    dttz = utc_dttz.astimezone(pytz.timezone(tz))

    return dttz


def dttz_to_ts(dttz):
    r"""Convert a datetime object into a timestamp.

    Parameters
    ----------
    dttz : datetime object
        Datetime object with a particular time zone.

    Returns
    -------
    ts: float
        The converted timestamp in seconds.
    """

    ts = datetime.timestamp(dttz)

    return ts
