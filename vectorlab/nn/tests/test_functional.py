import torch
import pytest

from vectorlab.nn.functional import kl_with_std_norm


@pytest.mark.parametrize('rerun', [100])
def test_kl_with_std_norm(rerun):

    for _ in range(rerun):

        mu, logstd = torch.rand([]), torch.rand([])
        l1 = kl_with_std_norm(mu, logstd, reduction='sum')

        p = torch.distributions.Normal(mu, logstd.exp())
        q = torch.distributions.Normal(0, 1)
        l2 = torch.distributions.kl_divergence(p, q)

        assert torch.isclose(l1, l2)
