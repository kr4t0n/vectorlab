import torch
import pytest

from vectorlab.nn.models._encoder import GRUEncoder
from vectorlab.nn.models._decoder import GRUDecoder
from vectorlab.nn.models._seq2seq import Seq2Seq


@pytest.mark.parametrize('n_words_1', [10, 20])
@pytest.mark.parametrize('n_words_2', [10, 20])
@pytest.mark.parametrize('emb_dims', [4])
@pytest.mark.parametrize('hidden_dims', [8])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('teacher_forcing', [0.0, 0.5, 1.0])
def test_seq2seq(n_words_1, n_words_2,
                 emb_dims, hidden_dims, num_layers,
                 teacher_forcing):

    x = torch.randint(0, n_words_1, (10, 4))
    y = torch.randint(0, n_words_2, (10, 4))

    net = Seq2Seq(
        encoder_embedding=torch.nn.Embedding(n_words_1, emb_dims),
        encoder=GRUEncoder(
            in_dims=emb_dims, hidden_dims=hidden_dims,
            num_layers=num_layers
        ),
        decoder_embedding=torch.nn.Embedding(n_words_2, emb_dims),
        decoder=GRUDecoder(
            in_dims=emb_dims, hidden_dims=hidden_dims, out_dims=n_words_2,
            num_layers=num_layers
        ),
        start_token=0,
        teacher_forcing=teacher_forcing
    )

    y_hat = net(x, y)

    assert y_hat.shape == (y.shape[0], y.shape[1], n_words_2)
