import time
import random
import pytest

from datetime import datetime, timedelta
from numpy.testing import assert_almost_equal

from vectorlab.utils import dt_to_dttz, ts_to_dttz, dttz_to_ts, get_real_date


@pytest.mark.repeat(10)
def test_dt_to_dttz():
    ts = time.time()

    utc_dt = datetime.utcfromtimestamp(ts)
    utc_dttz = dt_to_dttz(utc_dt, tz='UTC')
    utc_dttz_ = ts_to_dttz(ts, tz='UTC')

    assert utc_dttz == utc_dttz_


@pytest.mark.repeat(10)
def test_time_convert():
    ts = time.time()

    dttz = ts_to_dttz(ts)
    ts_ = dttz_to_ts(dttz)

    assert_almost_equal(ts, ts_, decimal=6)


@pytest.mark.repeat(10)
def test_get_real_date():
    date_str = '[DATE]'
    date_delta = random.randint(0, 10)
    date_plus_str = f'[DATE+{date_delta}]'
    date_minus_str = f'[DATE-{date_delta}]'

    assert get_real_date(date_str) == datetime.now().strftime('%Y%m%d')
    assert get_real_date(date_plus_str) == \
        (datetime.now() + timedelta(days=date_delta)).strftime('%Y%m%d')
    assert get_real_date(date_minus_str) == \
        (datetime.now() - timedelta(days=date_delta)).strftime('%Y%m%d')
