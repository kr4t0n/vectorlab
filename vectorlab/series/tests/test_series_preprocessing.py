import pytest
import numpy as np

from copy import deepcopy

from vectorlab.series import format_ts, series_interpolate


n_features = 10
n_samples = 10000

X = np.arange(n_samples)
Y = np.random.rand(n_features, n_samples)


@pytest.mark.parametrize('step', [10, 50, 100])
def test_format_ts(step):
    new_X, new_Y = format_ts(X, Y, step)

    assert new_Y.shape[0] == Y.shape[0]
    assert new_X.shape[0] == (X.shape[0] // step)
    assert new_X.shape[0] == new_Y.shape[1]


def format_ts_efficiency():
    format_ts(X, Y, step=100)


def test_format_ts_efficiency(benchmark):
    benchmark(format_ts_efficiency)


@pytest.mark.parametrize('kind',
                         [
                             'linear', 'nearest', 'nearest-up', 'zero',
                             'slinear', 'quadratic', 'cubic',
                             'previous', 'next'
                         ])
def test_series_interpolate(kind):
    masked_place = np.random.choice(range(n_samples), 10, replace=False)
    masked_Y = deepcopy(Y)
    masked_Y[:, masked_place] = np.nan

    new_Y = series_interpolate(X, masked_Y, kind=kind)

    assert new_Y.shape == Y.shape
    assert np.sum(np.isnan(new_Y)) == 0


def series_interpolate_efficiency():
    masked_place = np.random.choice(range(n_samples), 10, replace=False)
    masked_Y = deepcopy(Y)
    masked_Y[:, masked_place] = np.nan

    series_interpolate(X, masked_Y, kind='cubic')


def test_series_interpolate_efficiency(benchmark):
    benchmark(series_interpolate_efficiency)
