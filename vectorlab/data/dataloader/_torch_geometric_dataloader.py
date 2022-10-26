"""
Some useful Dataloaders specified for PyTorch Geometric training are proposed.
"""

import torch

from torch_geometric.data import Batch
from torch.utils.data import DataLoader

from ...utils._check import check_valid_option


def gnn_node_collate(batch):
    r"""Concatenate a graph into a graph.

    It is a dummy operation that may be useful in node
    level gnn training. It is used to fit into PyTorch data
    loader fashion in training.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        A single graph.

    Returns
    -------
    tuple
        A single graph.
    """

    return batch[0],


def gnn_graph_collate(batch):
    r"""Concatenate a batch of graphs into a large graph.

    We construct a large graph as a batch of data collated from
    small graphs, and return the batch data in a tuple form
    follow the torch tradition.

    Parameters
    ----------
    batch : list
        A single list of data to be collated.

    Returns
    -------
    tuple
        Tuple of a large graph.
    """

    batch = Batch.from_data_list(batch, None, None)

    return batch,


def _mask_gnn(data, p=0.5, mask_method='zeros'):
    r"""Mask GNN data node features.

    Randomly pick a portion of nodes in the GNN data, and mask their
    node features. Such masked features could be naively replaced by
    zeros or replaced by independent noise.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The GNN data object.
    p : float, optional
        The percentage nodes to be masked.
    mask_method : str, optional
        The method to replace the masked node features.

    Returns
    -------
    masked_data : torch_geometric.data.Data
        The masked GNN data object.
    """

    mask_method = check_valid_option(
        mask_method,
        options=['zeros', 'random'],
        variable_name='graph mask method'
    )

    masked_data = data.clone()
    num_nodes = masked_data.num_nodes

    mask = torch.rand(num_nodes)
    mask = (mask <= max(mask.min(), p))

    if mask_method == 'zeros':
        masked_data.x[mask] = 0
    elif mask_method == 'random':
        masked_data.x[mask] = torch.FloatTensor(
            masked_data.x[mask].shape
        ).uniform_(masked_data.x.min(), masked_data.x.max())

    masked_data.mask = mask

    return masked_data


def mask_gnn_collate(batch, p=0.5, mask_method='zeros'):
    r"""Concatenate a batch of graphs into a large graph and its
    corresponding masked graph.

    We construct a large graph as a batch of data collated from
    small graphs. And for each small graph, we also generate a
    corresponding masked version and also collated into a large
    masked graph. The large graph and masked one are returned
    together.

    Parameters
    ----------
    batch : list
        A single list of data to be collated.
    p : float, optional
        The percentage nodes to be masked.
    mask_method : str, optional
        The method to replace the masked node features.

    Returns
    -------
    tuple
        Tuple of a large graph and masked one.
    """

    masked_batch = [
        _mask_gnn(data, p=p, mask_method=mask_method)
        for data in batch
    ]

    batch = Batch.from_data_list(batch, None, None)
    masked_batch = Batch.from_data_list(masked_batch, None, None)

    return batch, masked_batch


class NodeDataLoader(DataLoader):
    r"""Load GNN data in a batch manner.

    NodeDataLoader inherits from original DataLoader,
    while using a customized collate function.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args, **kwargs,
            collate_fn=gnn_node_collate
        )

        return


class GraphDataLoader(DataLoader):
    r"""Load GNN data in a batch manner.

    GraphDataLoader inherits from original DataLoader,
    while using a customized collate function.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args, **kwargs,
            collate_fn=gnn_graph_collate
        )

        return


class MaskGraphDataLoader(DataLoader):
    r"""Load GNN data and masked data in a batch manner.

    MaskGraphDataLoader inherits from original Dataloader,
    while using a customized collate function to generate
    a batch of masked graphs with replacement.

    Parameters
    ----------
    p : float, optional
        The percentage nodes to be masked.
    mask_method : str, optional
        The method to replace the masked node features.
    """

    def __init__(self, *args, p=0.5, mask_method='zeros', **kwargs):

        super().__init__(
            *args, **kwargs,
            collate_fn=lambda batch: mask_gnn_collate(
                batch, p=p, mask_method=mask_method
            )
        )

        return
