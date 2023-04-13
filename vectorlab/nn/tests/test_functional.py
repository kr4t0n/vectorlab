import torch
import pytest

from vectorlab.nn.functional import kl_with_std_norm, graph_recon_loss


@pytest.mark.parametrize('rerun', [100])
def test_kl_with_std_norm(rerun):

    for _ in range(rerun):

        mu, logstd = torch.rand([]), torch.rand([])
        l1 = kl_with_std_norm(mu, logstd, reduction='sum')

        p = torch.distributions.Normal(mu, logstd.exp())
        q = torch.distributions.Normal(0, 1)
        l2 = torch.distributions.kl_divergence(p, q)

        assert torch.isclose(l1, l2, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('rerun', [100])
def test_graph_recon_loss(rerun):

    for _ in range(rerun):

        adj = torch.randint(0, 2, (10, 10))

        pos_edge_index = torch.where(adj == 1)
        neg_edge_index = torch.where(adj == 0)

        l = graph_recon_loss(adj, pos_edge_index, neg_edge_index)
        assert l.item() == 0
