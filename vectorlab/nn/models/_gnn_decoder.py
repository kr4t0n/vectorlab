import torch


class InnerProducetDecoder(torch.nn.Module):
    r"""An inner-product based decoder, defined as follows:

        .. math::
            \mathrm{DEC}(\mathbf{z}_u, \mathbf{z}_v) =
            \mathbf{z}_u^\top \mathbf{z}_v

    Here, we assume that the similarity between two nodes -
    e.g., the overlap between their local neighborhoods - is
    proportional to the dot product of their embeddings.

    Some examples of this style of node embedding algorithm
    includes the Graph Factorization (GF) approach, GraRep, and
    HOPE. All of three methods combine the inner-product decoder
    with the following mean-squared error:

        .. math::
            \mathcal{L} = \sum_{(u, v) \in \mathcal{D}} \|\mathrm{DEC}
            (\mathbf{z}_u, \mathbf{z}_v) - \mathbf{S}[u, v]\|_2^2

    They differ primarily in how they define :math:`\mathbf{S}[u, v]`,
    i.e., the notion of node-node similarity or neighborhood overlap
    that they use. Whereas the GF approach directly uses the adjacency
    matrix and sets :math:`\mathbf{S} \overset{\Delta}{=} \mathbf{A}`,
    the GraRep and HOPE approaches employ more general strategies. In
    particular, GraRep defines :math:`\mathbf{S}` based on the powers
    of the adjacency matrix, while the HOPE algorithm supports general
    neighborhood overlap measure.

    Parameters
    ----------
    sigmoid : bool
        Whether to use sigmoid function over the outputs or not.

    Attributes
    ----------
    sigmoid_ : bool
        Whether to use sigmoid function over the outputs or not.
    """

    def __init__(self, sigmoid=True):

        super().__init__()

        self.sigmoid_ = sigmoid

        return

    def forward(self, z, edge_index=None):
        r"""The forward process to obtain output adjacency matrix.

        In this function, it will decode latent space variable `z` into
        edge probabilities for provided node pairs, `edge_index`. If
        `edge_index` is not provided, all node pairs probabilities will
        be returned.

        Parameters
        ----------
        z : tensor
            The latent space input sample.
        edge_index : tensor, optional
            The edge index to be reconstructed.
        """

        if edge_index is None:
            adj = torch.matmul(z, z.T)
        else:
            adj = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        if self.sigmoid_:
            adj = torch.sigmoid(adj)

        return adj

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        return
