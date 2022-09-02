import pytest
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from vectorlab.optimize import extreme_estimation, slope_estimation

centers = [[100, 100], [-100, -100], [100, -100], [-100, 100]]
n_samples = 4000

X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.7)
criterion = 'inertia_'


@pytest.mark.parametrize('extreme_type', ['global_min', 'local_min'])
def test_extreme_estimation(extreme_type):

    lower_n_clusters, upper_n_clusters = 1, 10
    n_clusters = list(range(lower_n_clusters, upper_n_clusters + 1))
    kwargs_dict = {'n_clusters': n_clusters}

    kwargs, criteria = extreme_estimation(
        X, KMeans, 'fit', criterion,
        kwargs_dict, extreme_type=extreme_type,
        return_criteria=True
    )

    assert isinstance(kwargs, np.ndarray) and len(kwargs) == 1
    assert kwargs[0]['n_clusters'] == upper_n_clusters

    assert (
        isinstance(criteria, np.ndarray) and len(criteria) == len(n_clusters)
    )


def extreme_estimation_efficiency():

    lower_n_clusters, upper_n_clusters = 1, 10
    n_clusters = list(range(lower_n_clusters, upper_n_clusters + 1))
    kwargs_dict = {'n_clusters': n_clusters}

    extreme_estimation(
        X, KMeans, 'fit', criterion,
        kwargs_dict, extreme_type='global_min',
        return_criteria=True
    )


def test_extreme_estimation_efficiency(benchmark):
    benchmark(extreme_estimation_efficiency)


@pytest.mark.parametrize('slope_type', ['descending'])
def test_slope_estimation(slope_type):

    lower_n_clusters, upper_n_clusters = 1, 10
    n_clusters = list(range(lower_n_clusters, upper_n_clusters + 1))
    kwargs_dict = {'n_clusters': n_clusters}

    kwargs, criteria = slope_estimation(
        X, KMeans, 'fit', criterion,
        kwargs_dict, slope_type=slope_type,
        return_criteria=True
    )

    assert isinstance(kwargs, np.ndarray) and len(kwargs) == 1
    assert kwargs[0]['n_clusters'] == len(centers)

    assert (
        isinstance(criteria, np.ndarray) and len(criteria) == len(n_clusters)
    )


def slope_estimation_efficiency():

    lower_n_clusters, upper_n_clusters = 1, 10
    n_clusters = list(range(lower_n_clusters, upper_n_clusters + 1))
    kwargs_dict = {'n_clusters': n_clusters}

    slope_estimation(
        X, KMeans, 'fit', criterion,
        kwargs_dict, slope_type='descending',
        return_criteria=True
    )


def test_slope_estimation_efficiency(benchmark):
    benchmark(slope_estimation_efficiency)
