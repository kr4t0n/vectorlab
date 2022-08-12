import pytest
import numpy as np

from vectorlab.data import data_linear_generator


@pytest.mark.parametrize('n_dims', [8, 16, 128])
@pytest.mark.parametrize('n_samples', [10, 100, 1000])
@pytest.mark.parametrize('b', [0, 1e-8])
@pytest.mark.parametrize('eps', [0, 1e-8])
def test_data_linear_generator(n_dims, n_samples, b, eps):

    w = np.random.rand(n_dims)

    X, y = data_linear_generator(
        n_samples,
        w=w, b=b,
        noise=True, eps=eps
    )

    assert np.allclose(np.matmul(X, w) + b, y, atol=eps * 10)
