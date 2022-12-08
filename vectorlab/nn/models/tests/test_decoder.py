import torch
import pytest

from vectorlab.nn.models._decoder import (
    MLPDecoder,
    GRUDecoder, LSTMDecoder
)


@pytest.mark.parametrize('out_dims', [1, 4])
@pytest.mark.parametrize('num_layers', [1, 3])
@pytest.mark.parametrize('dropout', [.0, .5])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('sigmoid', [True, False])
@pytest.mark.parametrize('plain_last', [True, False])
def test_mlp_encoder(out_dims, num_layers, dropout, bias, sigmoid, plain_last):

    n_samples = 10
    hidden_dims = 2 * out_dims

    z = torch.rand(n_samples, hidden_dims)
    decoder = MLPDecoder(
        hidden_dims=hidden_dims, out_dims=out_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        sigmoid=sigmoid,
        plain_last=plain_last
    )

    x = decoder(z)
    assert x.shape == (n_samples, out_dims)


@pytest.mark.parametrize('in_dims', [1, 4])
@pytest.mark.parametrize('num_layers', [1, 3])
@pytest.mark.parametrize('dropout', [.0, .5])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('sigmoid', [True, False])
def test_rnn_encoder(in_dims, num_layers, dropout, bias, sigmoid):

    n_samples = 10
    n_seqs = 16
    hidden_dims = 2 * in_dims
    out_dims = in_dims

    X = torch.rand(n_seqs, n_samples, in_dims)
    h = torch.rand(num_layers, n_samples, hidden_dims)
    c = torch.rand(num_layers, n_samples, hidden_dims)

    gru_decoder = GRUDecoder(
        in_dims=in_dims, hidden_dims=hidden_dims, out_dims=out_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        sigmoid=sigmoid
    )
    lstm_decoder = LSTMDecoder(
        in_dims=in_dims, hidden_dims=hidden_dims, out_dims=out_dims,
        num_layers=num_layers,
        dropout=dropout,
        bias=bias,
        sigmoid=sigmoid
    )

    gru_output, gru_hidden = gru_decoder.forward(X, h)
    assert gru_output.shape == X.shape
    assert gru_hidden.shape == h.shape

    lstm_output, (lstm_hidden, lstm_cell) = lstm_decoder.forward(X, (h, c))
    assert lstm_output.shape == X.shape
    assert lstm_hidden.shape == h.shape
    assert lstm_cell.shape == c.shape
