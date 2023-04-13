import torch
import torch_geometric

from ._mlp import MLP
from ._basic_gnn import Graph


class _BasicEncoder(torch.nn.Module):
    r"""An abstract basic encoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def __init__(self, in_dims, hidden_dims, num_layers,
                 dropout=.5,
                 bias=True,
                 **kwargs):

        super().__init__()

        self.in_dims_ = in_dims
        self.hidden_dims_ = hidden_dims
        self.num_layers_ = num_layers
        self.dropout_ = dropout
        self.bias_ = bias
        self.kwargs_ = kwargs

        self.encoder_ = self._init_encoder()

        return

    def _init_encoder(self):
        r"""An abstract method to initialize desired encoder.

        Raises
        ------
        NotImplementedError
            When the method is not implemented, an error is raised.
        """

        raise NotImplementedError

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        self.encoder_.reset_parameters()

        return


class _BasicRNNEncoder(_BasicEncoder):
    r"""An abstract basic RNN encoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""An abstract method to initialize desired encoder.

        Raises
        ------
        NotImplementedError
            When the method is not implemented, an error is raised.
        """

        raise NotImplementedError

    def _flatten_hidden(self, h):
        r"""Flatten hidden state.

        Parameters
        ----------
        h : tensor
            The input hidden state.

        Returns
        -------
        tensor
            The flattened hidden state.
        """

        # flatten hidden states
        # (num_directions * num_layers, batch_size, out_dims) ==>
        # (batch_size, num_directions * num_layers, out_dims) ==>
        # (batch_size, num_directions * num_layers * out_dims)
        h = h.transpose(0, 1).contiguous().view(h.shape[1], -1)

        return h


class _BasicGNNEncoder(_BasicEncoder):
    r"""An abstract basic GNN encoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def forward(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        edge_index : tensor
            The input edge index.
        args : tuple
            The extra positional arguments for forward process.
        kwargs : dict
            The extra keyword arguments for forward process.

        Returns
        -------
        tensor
            The hidden samples.
        """

        x = self.encoder_(x, edge_index, *args, **kwargs)

        return x

    def forward_latent(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        edge_index : tensor
            The input edge index.
        args : tuple
            The extra positional arguments for forward process.
        kwargs : dict
            The extra keyword arguments for forward process.

        Returns
        -------
        tensor
            The latent samples.
        """

        x = self.encoder_(x, edge_index, *args, **kwargs)

        return x


class MLPEncoder(_BasicEncoder):
    r"""A MLP based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a MLP as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The MLP based encoder.
        """

        encoder = MLP(
            in_dims=self.in_dims_, hidden_dims=self.hidden_dims_,
            out_dims=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder

    def forward(self, x):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The hidden samples.
        """

        x = self.encoder_(x)

        return x

    def forward_latent(self, x):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The latent samples.
        """

        x = self.encoder_(x)

        return x


class GRUEncoder(_BasicRNNEncoder):
    r"""A GRU based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a GRU as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GRU based encoder.
        """

        encoder = torch.nn.GRU(
            input_size=self.in_dims_, hidden_size=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder

    def forward(self, x):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The hidden states.
        """

        _, h = self.encoder_(x)

        return h

    def forward_latent(self, x):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The latent samples.
        """

        _, h = self.encoder_(x)
        h = self._flatten_hidden(h)

        return h


class LSTMEncoder(_BasicRNNEncoder):
    r"""A LSTM based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a LSTM as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The LSTM based encoder.
        """

        encoder = torch.nn.LSTM(
            input_size=self.in_dims_, hidden_size=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder

    def forward(self, x):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tuple
            The hidden and cell states.
        """

        _, h = self.encoder_(x)

        return h

    def forward_latent(self, x):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The latent samples.
        """

        _, (h, c) = self.encoder_(x)

        # flatten hidden and cell states
        h = self._flatten_hidden(h)
        c = self._flatten_hidden(c)

        h = torch.cat((h, c), dim=1)

        return h


class GCNEncoder(_BasicGNNEncoder):
    r"""A GCNConv based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a GCN as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GCN based encoder.
        """

        encoder = torch_geometric.nn.models.GCN(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class GraphEncoder(_BasicGNNEncoder):
    r"""A GraphConv based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a Graph as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The Graph based encoder.
        """

        encoder = Graph(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class GATEncoder(_BasicGNNEncoder):
    r"""A GATConv based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a GAT as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GAT based encoder.
        """

        encoder = torch_geometric.nn.models.GAT(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class GINEncoder(_BasicGNNEncoder):
    r"""A GINConv based encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    """

    def _init_encoder(self):
        r"""Initialize a GIN as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GIN based encoder.
        """

        encoder = torch_geometric.nn.models.GIN(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class _BasicVarEncoder(_BasicEncoder):
    r"""An abstract basic variational encoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def __init__(self, in_dims, hidden_dims, num_layers,
                 dropout=.5,
                 bias=True,
                 **kwargs):

        super().__init__(
            in_dims, hidden_dims, num_layers,
            dropout=dropout,
            bias=bias,
            **kwargs
        )

        self.mu_ = torch.nn.Linear(
            in_features=self.hidden_dims_, out_features=self.hidden_dims_,
            bias=self.bias_
        )
        self.logstd_ = torch.nn.Linear(
            in_features=self.hidden_dims_, out_features=self.hidden_dims_,
            bias=self.bias_
        )

        return

    def _init_encoder(self):
        r"""An abstract method to initialize desired encoder.

        Raises
        ------
        NotImplementedError
            When the method is not implemented, an error is raised.
        """

        raise NotImplementedError

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        self.encoder_.reset_parameters()
        self.mu_.reset_parameters()
        self.logstd_.reset_parameters()

        return


class _BasicVarGNNEncoder(_BasicVarEncoder):
    r"""An abstract basic GNN variational encoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def forward(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        edge_index : tensor
            The input edge index.
        args : tuple
            The extra positional arguments for forward process.
        kwargs : dict
            The extra keyword arguments for forward process.

        Returns
        -------
        mu : tensor
            The output mu value of samples.
        logstd : tensor
            The output logstd value of samples.
        """

        x = self.encoder_(x, edge_index, *args, **kwargs)

        mu = self.mu_(x)
        logstd = self.logstd_(x)

        return mu, logstd


class VarMLPEncoder(_BasicVarEncoder):
    r"""An abstract basic variational encoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def _init_encoder(self):
        r"""Initialize a MLP as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The MLP based encoder.
        """

        encoder = MLP(
            in_dims=self.in_dims_, hidden_dims=self.hidden_dims_,
            out_dims=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder

    def forward(self, x):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        mu : tensor
            The output mu value of samples.
        logstd : tensor
            The output logstd value of samples.
        """

        x = self.encoder_(x)

        mu = self.mu_(x)
        logstd = self.logstd_(x)

        return mu, logstd


class VarGCNEncoder(_BasicVarGNNEncoder):
    r"""An GCNConv based variational encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def _init_encoder(self):
        r"""Initialize a GCN as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GCN based encoder.
        """

        encoder = torch_geometric.nn.models.GCN(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class VarGraphEncoder(_BasicVarGNNEncoder):
    r"""An GraphConv based variational encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def _init_encoder(self):
        r"""Initialize a Graph as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The Graph based encoder.
        """

        encoder = Graph(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class VarGATEncoder(_BasicVarGNNEncoder):
    r"""An GATConv based variational encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def _init_encoder(self):
        r"""Initialize a GAT as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GAT based encoder.
        """

        encoder = torch_geometric.nn.models.GAT(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder


class VarGINEncoder(_BasicVarGNNEncoder):
    r"""An GINConv based variational encoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    kwargs : dict
        The extra arguments passed to initialize an encoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    kwargs_ : dict
        The extra arguments passed to initialize an encoder.
    encoder_ : torch.nn.Module
        The initialized encoder.
    mu_ : torch.nn.Linear
        The linear layer to obtain mu value.
    logstd_ : torch.nn.Linear
        The linear layer to obtain logstd value.
    """

    def _init_encoder(self):
        r"""Initialize a GIN as the encoder.

        Returns
        -------
        encoder : torch.nn.Module
            The GIN based encoder.
        """

        encoder = torch_geometric.nn.models.GIN(
            in_channels=self.in_dims_, hidden_channels=self.hidden_dims_,
            out_channels=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return encoder
