"""
Useful Datasets specified for PyTorch training.
"""

import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from ...base import SLMixin
from ...utils._check import _check_tensor


class MultiDataset(SLMixin, Dataset):
    r"""MultiDataset is a data structure to store multiple PyTorch data
    sources.

    MultiDataset aims to feed the data in batch for multiple data sources.
    A condition for using this data structure is that all these data sources
    should have the same first dimension.

    Parameters
    ----------
    args : tuple
        The tuple of multiple data inputs.

    Attributes
    ----------
    len_ : int
        The number of data stored in MultiDataset.
    tensors_ : tuple
        The multiple tensors stored in MultiDataset.
    """

    def __init__(self, *args):

        super().__init__()

        tensors = tuple(_check_tensor(arg) for arg in args)

        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), 'Size mismatch between tensors.'

        self.len_ = tensors[0].size(0)
        self.tensors_ = tensors

        return

    def __len__(self):
        r"""Return the number of data stored in MultiDataset.

        Returns
        -------
        int
            The number of data stored in MultiDataset.
        """

        return self.len_

    def __getitem__(self, idx):
        r"""Fetch the data using idx parameter.

        Since MultiDataset stored data from multiple data sources, as a
        result we use a slice to fetch data separately, and zip them as
        a list.

        Parameters
        ----------
        idx : slice
            The slice used to fetch the data.

        Returns
        -------
        list
            The list of data from multiple data sources fetched by slice.
        """

        return tuple(tensor[idx] for tensor in self.tensors_)


class ImageDataset(SLMixin, Dataset):
    r"""ImageDataset is a data structure to store image data sources.

    Given a root directory of multiple images and an annotation file
    stored image names in the first column, and corresponding labels
    in the second column. ImageDataset aims to feed the data in batch
    for image data sources.

    Parameters
    ----------
    root : str
        The root directory for images.
    annotation : pd.DataFrame
        The DataFrame stored the images related information.
    transform : torch.nn.Module
        A series of transform(s) applied over the given image(s).

    Attributes
    ----------
    root_ : str
        The root directory for images.
    annotations_ : pd.DataFrame
        The DataFrame stored the images related information.
    transform_ : torch.nn.Module
        A series of transform(s) applied over the given image(s).
    len_ : int
        The number of data stored in ImageDataset.
    """

    def __init__(self, root, annotation, transform=None):

        super().__init__()

        self.root_ = root
        self.annotations_ = pd.read_csv(annotation)
        self.transform_ = transform

        self.len_ = len(self.annotations_)

        return

    def __len__(self):
        r"""Return the number of data stored in ImageDataset.

        Returns
        -------
        int
            The number of data stored in ImageDataset.
        """

        return self.len_

    def __getitem__(self, idx):
        r"""Fetch the data using idx parameter.

        Since ImageDataset stored the image related information, it uses
        idx to fetch the corresponding image name and label from the
        annotation, while the image is loaded using OpenCV. If the a
        certain transform is provided, it will be used once the image is
        loaded.

        Parameters
        ----------
        idx : slice
            The slice used to fetch the data.

        Returns
        -------
        tuple
            The tuple of data contained the image and label.
        """

        img_path = os.path.join(
            self.root_,
            self.annotations_.iloc[idx, 0]
        )

        img = Image.open(img_path)
        img = img.convert('RGB')

        if self.transform_:
            img = self.transform_(img)

        label = _check_tensor(
            int(self.annotations_.iloc[idx, 1])
        )

        return img, label


class SequenceDataset(SLMixin, Dataset):
    r"""SequenceDataset is a data structure to store sequence data.

    Given a series of input sequences and output sequences, this data
    structure could store the corresponding sequences. The number of
    input sequences and output sequences should be the same, while
    the length inside input sequences and output sequences could vary.

    Parameters
    ----------
    input_seq : list or tuple of tensors
        The input sequences.
    output_seq : list or tuple of tensors
        The output sequences.

    Attributes
    ----------
    input_seq_ : list or tuple of tensors
        The input sequences.
    output_seq_ : list or tuple of tensors
        The output sequences.
    len_ : int
        The number of sequences stored in SequenceDataset.
    """

    def __init__(self, input_seq, output_seq):

        super().__init__()

        assert len(input_seq) == len(output_seq)

        self.input_seq_ = tuple(_check_tensor(_) for _ in input_seq)
        self.output_seq_ = tuple(_check_tensor(_) for _ in output_seq)

        self.len_ = len(self.input_seq_)

        return

    def __len__(self):
        r"""Return the number of sequences stored in SequenceDataset.

        Returns
        -------
        int
            The number of data stored in SequenceDataset.
        """

        return self.len_

    def __getitem__(self, idx):
        r"""Fetch the data using idx parameter.

        We use a slice to fetch input sequences and output sequences
        separately.

        Parameters
        ----------
        idx : slice
            The slice used to fetch the data.

        Returns
        -------
        tuple
            The tuple of data contained the input and output sequences.
        """

        return self.input_seq_[idx], self.output_seq_[idx]
