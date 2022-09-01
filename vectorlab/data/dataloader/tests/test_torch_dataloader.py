import torch
import random
import pytest

from vectorlab.data.dataset import SequenceDataset
from vectorlab.data.dataloader import (
    PadSeqDataLoader,
    PadSeqsDataLoader
)

n_samples = 10
n_lens = [random.randint(0, 10) for _ in range(n_samples)]
batch_size = 10


@pytest.mark.parametrize('batch_first', [True, False])
def test_PadSeqDataLoader(batch_first):

    input_seq = [torch.rand(n_len) for n_len in n_lens]
    output = torch.rand(n_samples)
    max_len = max(n_lens)

    dataset = SequenceDataset(input_seq, output)
    loader = PadSeqDataLoader(
        dataset=dataset, batch_first=batch_first, batch_size=batch_size
    )

    X, X_lens, y = next(iter(loader))

    if batch_first:
        assert X.shape == (batch_size, max_len)
    else:
        assert X.shape == (max_len, batch_size)

    assert all(X_lens == torch.tensor(n_lens))
    assert y.shape == (batch_size, )


@pytest.mark.parametrize('batch_first', [True, False])
def test_PadSeqsDataLoader(batch_first):

    input_seq = [torch.rand(n_len) for n_len in n_lens]
    output_seq = [torch.rand(n_len) for n_len in n_lens]
    max_len = max(n_lens)

    dataset = SequenceDataset(input_seq, output_seq)
    loader = PadSeqsDataLoader(
        dataset=dataset, batch_first=batch_first, batch_size=batch_size
    )

    X, X_lens, y, y_lens = next(iter(loader))

    if batch_first:
        assert X.shape == (batch_size, max_len)
        assert y.shape == (batch_size, max_len)
    else:
        assert X.shape == (max_len, batch_size)
        assert y.shape == (max_len, batch_size)

    assert all(X_lens == torch.tensor(n_lens))
    assert all(y_lens == torch.tensor(n_lens))
