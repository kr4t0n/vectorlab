import torch
import pytest

from vectorlab.nn.models._encoder import (
    MLPEncoder,
    GRUEncoder, LSTMEncoder,
    VarMLPEncoder
)


@pytest.mark.parametrize('in_dims', [1, 4])
@pytest.mark.parametrize('num_layers', [1, 3])
@pytest.mark.parametrize('dropout', [.0, .5])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('plain_last', [True, False])
def test_mlp_encoder(in_dims, num_layers, bias, dropout, plain_last):

    n_samples = 10
    hidden_dims = 2 * in_dims

    X = torch.rand(n_samples, in_dims)
    encoder = MLPEncoder(
        in_dims=in_dims, hidden_dims=hidden_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        plain_last=plain_last
    )
    var_encoder = VarMLPEncoder(
        in_dims=in_dims, hidden_dims=hidden_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        plain_last=plain_last
    )

    z = encoder(X)
    assert z.shape == (n_samples, hidden_dims)

    mu, logstd = var_encoder(X)
    assert mu.shape == (n_samples, hidden_dims)
    assert logstd.shape == (n_samples, hidden_dims)


@pytest.mark.parametrize('in_dims', [1, 4])
@pytest.mark.parametrize('num_layers', [1, 3])
@pytest.mark.parametrize('dropout', [.0, .5])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('bidirectional', [True, False])
def test_rnn_encoder(in_dims, num_layers, bias, dropout, bidirectional):

    n_samples = 10
    n_seqs = 16
    hidden_dims = 2 * in_dims

    if bidirectional:
        d = 2
    else:
        d = 1

    X = torch.rand(n_seqs, n_samples, in_dims)
    gru_encoder = GRUEncoder(
        in_dims=in_dims, hidden_dims=hidden_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        bidirectional=bidirectional
    )
    lstm_encoder = LSTMEncoder(
        in_dims=in_dims, hidden_dims=hidden_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        bidirectional=bidirectional
    )

    gru_hidden = gru_encoder.forward_latent(X)
    assert gru_hidden.shape == (n_samples, d * num_layers * hidden_dims)

    lstm_hidden = lstm_encoder.forward_latent(X)
    assert lstm_hidden.shape == (n_samples, 2 * d * num_layers * hidden_dims)
