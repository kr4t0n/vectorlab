import torch
import pytest

from vectorlab.nn.models._decoder import (
    MLPDecoder
)


@pytest.mark.parametrize('out_dims', [1, 4])
@pytest.mark.parametrize('num_layers', [1, 3])
@pytest.mark.parametrize('dropout', [.0, .5])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('plain_last', [True, False])
def test_mlp_encoder(out_dims, num_layers, bias, dropout, plain_last):

    n_samples = 10
    hidden_dims = 2 * out_dims

    z = torch.rand(n_samples, hidden_dims)
    decoder = MLPDecoder(
        hidden_dims=hidden_dims, out_dims=out_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        plain_last=plain_last
    )

    x = decoder(z)
    assert x.shape == (n_samples, out_dims)
