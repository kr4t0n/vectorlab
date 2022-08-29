import pytest
import numpy as np

from vectorlab.metrics.pairwise import (
    dtw, dtw_estimate,
    lp_norm,
    sbd
)

n_samples = 100
n_features = 3

X = np.random.rand(n_samples, n_features)
Y = np.random.rand(n_samples, n_features)


@pytest.mark.parametrize('weighted', [True, False])
def test_dtw(weighted):

    # test identity of indiscernibles
    assert np.isclose(dtw(X, X, weighted=weighted), 0)
    assert np.isclose(dtw(Y, Y, weighted=weighted), 0)

    # test symmetry
    assert np.isclose(
        dtw(X, Y, weighted=weighted),
        dtw(Y, X, weighted=weighted)
    )


def dtw_efficency():
    dtw(X, Y)


def test_dtw_efficency(benchmark):
    benchmark(dtw_efficency)


@pytest.mark.parametrize('method', ['LB_Kim', 'LB_Keogh', 'LB_Keogh_Reversed'])
def test_dtw_estimate(method):

    assert dtw_estimate(X, Y, method=method) <= dtw(X, Y)


def dtw_estimate_efficency(method):
    dtw_estimate(X, Y, method=method)


@pytest.mark.parametrize('method', ['LB_Kim', 'LB_Keogh', 'LB_Keogh_Reversed'])
def test_dtw_estimate_efficency(method, benchmark):
    benchmark.pedantic(
        dtw_estimate_efficency,
        kwargs={'method': method},
        rounds=100
    )


def test_lp_norm():

    # test identity of indiscernibles
    assert np.isclose(lp_norm(X, X), 0)
    assert np.isclose(lp_norm(Y, Y), 0)

    # test symmetry
    assert np.isclose(lp_norm(X, Y), lp_norm(Y, X))


def lp_norm_efficency():
    lp_norm(X, Y)


def test_lp_norm_efficency(benchmark):
    benchmark(lp_norm_efficency)


@pytest.mark.parametrize('shifting', [True, False])
def test_sbd(shifting):

    # test identity of indiscernibles
    assert np.isclose(sbd(X, X, shifting=shifting), 0)
    assert np.isclose(sbd(Y, Y, shifting=shifting), 0)

    # test symmetry
    assert np.isclose(
        sbd(X, Y, shifting=shifting),
        sbd(Y, X, shifting=shifting)
    )


def sbd_efficency():
    sbd(X, Y)


def test_sbd_efficency(benchmark):
    benchmark(sbd_efficency)
