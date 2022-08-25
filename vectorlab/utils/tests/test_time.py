import time
import pytest

from datetime import datetime
from numpy.testing import assert_almost_equal

from vectorlab.utils import dt_to_dttz, ts_to_dttz, dttz_to_ts


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
