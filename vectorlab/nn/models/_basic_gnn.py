from torch_geometric.nn import GraphConv
from torch_geometric.nn.models.basic_gnn import BasicGNN


class Graph(BasicGNN):
    r"""The graph neural network from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"

    Parameters
    ----------
    in_channels : int, tuple
        Size of each input sample, a tuple correponds to the sizes
        of source and target dimensionalities.
    hidden_channles : int
        Size of each hidden sample.
    num_layers : int
        Number of message passing layers.
    out_channles : int, optional
        If not set to None, will apply a final linear transformation
        to convert hidden node embeddings to output size.
    dropout : float, optional
        Dropout probability.
    act : str, callable, optional
        The non-linear activation function to use.
    act_first : bool, optional
        If set to True, acitvation is applied before normalization.
    act_kwargs : dict, optional
        Arguments passed to the respective activation function
        defined by act.
    norm : str, callable, optional
        The normalizaiton function to use.
    norm_kwargs : dict, optional
        Arguments passed to the respective normalization function
        defined by norm.
    jk : str, optional
        The Jumping Knowledge mode. If specified, the model will
        additionally apply a final linear transformation to transform
        node embeddings to the expect output feature dimensionality.
    kwargs: dict, optional
        Additional arguments of GraphConv.
    """

    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, in_channels, out_channels, **kwargs):
        r"""initialize the convolutional layer.

        Parameters
        ----------
        in_channels : int, tuple
            Size of each input sample, a tuple correponds to the sizes
            of source and target dimensionalities.
        out_channles : int, optional
            If not set to None, will apply a final linear transformation
            to convert hidden node embeddings to output size.

        Returns
        -------
        torch_geometric.nn.conv.GraphConv
            The initialized GraphConv layer.
        """

        return GraphConv(in_channels, out_channels, **kwargs)
