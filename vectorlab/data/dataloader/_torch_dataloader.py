"""
Some useful Dataloaders specified for PyTorch training are proposed.
"""

import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate


def pad_sequence_collate(batch, batch_first=False):
    r"""Padding a batch of sequence data

    The element inside such batch containing a sequence and a
    desired target. Therefore, we only want to pad the sequence
    part, X. This collate function will return the padded
    sequence, original sequence length and a batch of targets.

    Parameters
    ----------
    batch : tuple
        A single batch to be collated.
    batch_first : bool, optional
        If batch_first is True, the number of first dimension is batch size,
        otherwise, the batch size will be the second dimension.

    Returns
    -------
    tuple
        Tuple of padded sequence, original length and targets.
    """

    (X, y) = zip(*batch)

    X_lens = torch.tensor([len(x_) for x_ in X], dtype=torch.long)
    X = pad_sequence(X, batch_first=batch_first)

    y = default_collate(y)

    return X, X_lens, y


def pad_sequences_collate(batch, batch_first=False):
    r"""Padding a batch of sequences data

    The element inside such batch containing a input sequence and a
    output sequence. Therefore, we not only want to pad the input
    sequence, X, but also want to pad the output sequence, y.
    This collate function will return the padded input sequence,
    original input sequence length, padded output sequence and
    original output sequence length.

    Parameters
    ----------
    batch : tuple
        A single batch to be collated.
    batch_first : bool, optional
        If batch_first is True, the number of first dimension is batch size,
        otherwise, the batch size will be the second dimension.

    Returns
    -------
    tuple
        Tuple of padded input sequence, original input sequence length,
        padded output sequence and original output sequence length.
    """

    (X, y) = zip(*batch)

    X_lens = torch.tensor([len(x_) for x_ in X], dtype=torch.long)
    X = pad_sequence(X, batch_first=batch_first)

    y_lens = torch.tensor([len(y_) for y_ in y], dtype=torch.long)
    y = pad_sequence(y, batch_first=batch_first)

    return X, X_lens, y, y_lens


class PadSeqDataLoader(DataLoader):
    r"""Load data in a padding manner.

    PadSeqDataLoader inherits from original Dataloader,
    while using a customized pad sequence collate function.

    Parameters
    ----------
    batch_first : bool
        Whether pad sequence in a batch first manner or not.

    Attributes
    ----------
    batch_first : bool
        Whether pad sequence in a batch first manner or not.
    """

    def __init__(self, *args, batch_first=False, **kwargs):

        super().__init__(
            *args, **kwargs,
            collate_fn=lambda batch: pad_sequence_collate(
                batch, batch_first=batch_first
            )
        )

        self.batch_first = batch_first

        return


class PadSeqsDataLoader(DataLoader):
    r"""Load data in a padding manner.

    PadSeqsDataLoader inherits from original Dataloader,
    while using a customized pad sequence collate function.

    Parameters
    ----------
    batch_first : bool
        Whether pad sequence in a batch first manner or not.

    Attributes
    ----------
    batch_first : bool
        Whether pad sequence in a batch first manner or not.
    """

    def __init__(self, *args, batch_first=False, **kwargs):

        super().__init__(
            *args, **kwargs,
            collate_fn=lambda batch: pad_sequences_collate(
                batch, batch_first=batch_first
            )
        )

        self.batch_first = batch_first

        return
