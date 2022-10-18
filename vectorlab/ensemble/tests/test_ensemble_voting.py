import pytest
import numpy as np

from vectorlab.ensemble._voting import voting, _soft_voting, _hard_voting


@pytest.mark.parametrize('n_samples', [10, 100])
@pytest.mark.parametrize('n_classes', [3, 10])
@pytest.mark.parametrize('weights', [None, [1, 0, 0], [0, 1, 0], [0, 0, 1]])
@pytest.mark.parametrize('method', ['soft', 'hard'])
def test_voting(n_samples, n_classes, weights, method):

    a = np.random.rand(n_samples, n_classes)
    b = np.random.rand(n_samples, n_classes)
    c = np.random.rand(n_samples, n_classes)

    if method == 'soft':
        assert np.all(
            voting(
                (a, b, c),
                weights=weights,
                method=method
            ) == _soft_voting(
                (a, b, c),
                weights=weights
            )
        )
    elif method == 'hard':
        ad = np.argmax(a, axis=1)
        bd = np.argmax(b, axis=1)
        cd = np.argmax(c, axis=1)

        assert np.all(
            voting(
                (a, b, c),
                weights=weights,
                method=method
            ) == _hard_voting(
                (ad, bd, cd),
                weights=weights
            )
        )


def voting_soft_efficency():
    n_samples = 100
    n_classes = 10

    a = np.random.rand(n_samples, n_classes)
    b = np.random.rand(n_samples, n_classes)
    c = np.random.rand(n_samples, n_classes)

    voting((a, b, c), weights=None, method='soft')


def test_voting_soft_efficency(benchmark):
    benchmark(voting_soft_efficency)


def voting_hard_efficency():
    n_samples = 100
    n_classes = 10

    a = np.random.rand(n_samples, n_classes)
    b = np.random.rand(n_samples, n_classes)
    c = np.random.rand(n_samples, n_classes)

    voting((a, b, c), weights=None, method='hard')


def test_voting_hard_efficency(benchmark):
    benchmark(voting_hard_efficency)


@pytest.mark.parametrize('n_samples', [10, 100])
@pytest.mark.parametrize('n_classes', [3, 10])
@pytest.mark.parametrize('weights', [None, [1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_soft_voting(n_samples, n_classes, weights):

    a = np.random.rand(n_samples, n_classes)
    b = np.random.rand(n_samples, n_classes)
    c = np.random.rand(n_samples, n_classes)

    if weights is None:
        assert np.all(
            _soft_voting(
                (a, b, c),
                weights=weights
            ) == np.argmax(
                a + b + c,
                axis=1
            )
        )
    else:
        assert np.all(
            _soft_voting(
                (a, b, c),
                weights=weights
            ) == np.argmax(
                a * weights[0] + b * weights[1] + c * weights[2],
                axis=1
            )
        )


def soft_voting_efficency():
    n_samples = 100
    n_classes = 10

    a = np.random.rand(n_samples, n_classes)
    b = np.random.rand(n_samples, n_classes)
    c = np.random.rand(n_samples, n_classes)

    _soft_voting((a, b, c), weights=None)


def test_soft_voting_efficency(benchmark):
    benchmark(soft_voting_efficency)


@pytest.mark.parametrize('n_samples', [10, 100])
@pytest.mark.parametrize('n_classes', [3, 10])
@pytest.mark.parametrize('weights', [None, [1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_hard_voting(n_samples, n_classes, weights):

    a = np.random.randint(0, n_classes + 1, n_samples)
    b = np.random.randint(0, n_classes + 1, n_samples)
    c = np.random.randint(0, n_classes + 1, n_samples)

    assert np.all(
        _hard_voting(
            (a, b, c),
            weights=weights
        ) == np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=weights)),
            axis=0,
            arr=np.stack((a, b, c))
        )
    )


def hard_voting_efficency():
    n_samples = 100
    n_classes = 10

    a = np.random.randint(0, n_classes + 1, n_samples)
    b = np.random.randint(0, n_classes + 1, n_samples)
    c = np.random.randint(0, n_classes + 1, n_samples)

    _hard_voting((a, b, c), weights=None)


def test_hard_voting_efficency(benchmark):
    benchmark(hard_voting_efficency)
