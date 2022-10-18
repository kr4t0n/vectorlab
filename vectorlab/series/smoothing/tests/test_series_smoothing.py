import pytest
import numpy as np

from vectorlab.series.smoothing import (
    MovingAverage,
    WeightedMovingAverage,
    ExpWeightedMovingAverage,
    ARIMA,
    HoltWinters,
    SpectralResidual
)

n_samples = 10000
X = np.random.rand(n_samples)


@pytest.mark.parametrize('window_size', [3, 10, 50])
def test_ma(window_size):
    ma = MovingAverage(window_size=window_size)
    X_ma = ma.fit_transform(X)

    assert X.shape == X_ma.shape


def ma_efficency():
    ma = MovingAverage(window_size=3)
    ma.fit_transform(X)


def test_ma_efficency(benchmark):
    benchmark(ma_efficency)


@pytest.mark.parametrize('window_size', [3, 10, 50])
def test_wma(window_size):
    wma = WeightedMovingAverage(window_size=window_size)
    X_wma = wma.fit_transform(X)

    assert X.shape == X_wma.shape


def wma_efficency():
    wma = WeightedMovingAverage(window_size=3)
    wma.fit_transform(X)


def test_wma_efficency(benchmark):
    benchmark(wma_efficency)


@pytest.mark.parametrize('alpha', [0.1, 0.5, 0.7])
def test_ewma(alpha):
    ewma = ExpWeightedMovingAverage(alpha=0.5)
    X_ewma = ewma.fit_transform(X)

    assert X.shape == X_ewma.shape


def ewma_efficency():
    ewma = ExpWeightedMovingAverage(alpha=0.5)
    ewma.fit_transform(X)


def test_ewma_efficency(benchmark):
    benchmark(ewma_efficency)


@pytest.mark.parametrize('order', ['auto'])
def test_arima(order):
    arima = ARIMA(order=order)
    X_arima = arima.fit_transform(X)

    assert X.shape == X_arima.shape


def arima_efficency():
    arima = ARIMA(order='auto')
    arima.fit_transform(X)


def test_arima_efficency(benchmark):
    benchmark(arima_efficency)


def test_holt_winters():
    holt_winters = HoltWinters()
    X_holt_winters = holt_winters.fit_transform(X)

    assert X.shape == X_holt_winters.shape


def holt_winters_efficency():
    holt_winters = HoltWinters()
    holt_winters.fit_transform(X)


def test_holt_winters_efficency(benchmark):
    benchmark(holt_winters_efficency)


@pytest.mark.parametrize('window_size', [3, 10, 50])
def test_sr(window_size):
    sr = SpectralResidual(window_size=window_size)
    X_sr = sr.fit_transform(X)

    assert X.shape == X_sr.shape


def sr_efficency():
    sr = SpectralResidual(window_size=3)
    sr.fit_transform(X)


def test_sr_efficency(benchmark):
    benchmark(sr_efficency)
