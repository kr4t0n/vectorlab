import torch

from .. import functional as F
from torch_geometric.utils import negative_sampling


class GraphReconLoss(torch.nn.modules.loss._Loss):
    r"""Graph reconstruction loss.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output.
    """

    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):

        super().__init__(reduction=reduction)

        return

    def forward(self, adj, pos_edge_index, neg_edge_index=None):
        r"""The forward process to obtain loss result.

        Input adjacency matrix stands for edge probabilities of each
        pair of nodes. Positive edge index stands for the edges
        appeared in the original graph, while negative edge index is
        used as a panelty for the edges are not inside the original
        graph.

        Parameters
        ----------
        adj : tensor
            The adjacency matrix of reconstructed graph.
        pos_edge_index : tensor
            The positive edge index.
        neg_edge_index : tensor, optional
            The negative edge index.

        Returns
        -------
        loss : tensor
            The graph reconstruction loss.
        """

        if neg_edge_index is None:
            n_nodes = adj.shape[0]
            neg_edge_index = negative_sampling(pos_edge_index, n_nodes)

        loss = F.graph_recon_loss(
            adj, pos_edge_index, neg_edge_index,
            reduction=self.reduction
        )

        return loss
