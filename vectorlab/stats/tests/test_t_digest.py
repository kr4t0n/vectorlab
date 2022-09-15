from unittest import result
import pytest
import numpy as np

from vectorlab.stats import TDigest

T_DIGEST_PKL = 't_digest.pkl'

n_samples = 10000
x = np.random.rand(n_samples)


@pytest.mark.parametrize('quantiles', [np.array([0.25, 0.5, 0.75])])
def test_t_digest(quantiles):

    t_digest = TDigest(buffer_size=100)
    t_digest.fit(x)

    results = t_digest.predict(quantiles)
    actual_results = np.sort(x)[(quantiles * n_samples).astype(np.int_)]

    mse = np.mean((results - actual_results) ** 2)
    print(f'MSE: {mse}')

    assert t_digest.min_ == x.min()
    assert t_digest.max_ == x.max()
    assert t_digest.total_num_ == n_samples
    assert t_digest.predict([0.0])[0] == x.min()
    assert t_digest.predict([1.0])[0] == x.max()

    TDigest.save(t_digest, T_DIGEST_PKL)
    loaded_t_digest = TDigest.load(T_DIGEST_PKL)

    assert loaded_t_digest.buffer_ == t_digest.buffer_
    assert loaded_t_digest.buffer_size_ == t_digest.buffer_size_
    assert loaded_t_digest.clusters_ == t_digest.clusters_
    assert loaded_t_digest.min_ == t_digest.min_
    assert loaded_t_digest.max_ == t_digest.max_
    assert loaded_t_digest.total_num_ == t_digest.total_num_
    assert (loaded_t_digest.quantiles_ == t_digest.quantiles_).all()
    assert (loaded_t_digest.values_ == t_digest.values_).all()
