import torch
import warnings

from ._mlp import MLP


class _BasicDecoder(torch.nn.Module):
    r"""An abstract basic decoder class.

    Parameters
    ----------
    hidden_dims : int
        The dimension of hidden sample.
    out_dims : int
        The dimension of output sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    sigmoid : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs : dict
        The extra arguments passed to initialize an decoder.

    Attributes
    ----------
    hidden_dims_ : int
        The dimension of hidden sample.
    out_dims_ : int
        The dimension of output sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    sigmoid_ : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs_ : dict
        The extra arguments passed to initialize an decoder.
    decoder_ : torch.nn.Module
        The initialized decoder.
    """

    def __init__(self, hidden_dims, out_dims, num_layers,
                 dropout=.5,
                 bias=True,
                 sigmoid=True,
                 **kwargs):

        super().__init__()

        self.hidden_dims_ = hidden_dims
        self.out_dims_ = out_dims
        self.num_layers_ = num_layers
        self.dropout_ = dropout
        self.bias_ = bias
        self.sigmoid_ = sigmoid
        self.kwargs_ = kwargs

        self.decoder_ = self._init_decoder()

        return

    def _init_decoder(self):
        r"""An abstract method to initialize decoder encoder.

        Raises
        ------
        NotImplementedError
            When the method is not implemented, an error is raised.
        """

        raise NotImplementedError

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        self.decoder_.reset_parameters()

        return


class _BasicRNNDecoder(_BasicDecoder):
    r"""An abstract basic RNN decoder class.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    out_dims : int
        The dimension of output sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    sigmoid : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs : dict
        The extra arguments passed to initialize an decoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    out_dims_ : int
        The dimension of output sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    sigmoid_ : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs_ : dict
        The extra arguments passed to initialize an decoder.
    decoder_ : torch.nn.Module
        The initialized decoder.
    out_ : torch.nn.Module
        The output layer.
    """

    def __init__(self, in_dims, hidden_dims, out_dims, num_layers,
                 dropout=.5,
                 bias=True,
                 sigmoid=True,
                 **kwargs):

        self.in_dims_ = in_dims

        super().__init__(hidden_dims, out_dims, num_layers,
                         dropout=dropout,
                         bias=bias,
                         sigmoid=sigmoid,
                         **kwargs)

        self.out_ = torch.nn.Linear(
            in_features=self.hidden_dims_, out_features=self.out_dims_,
            bias=self.bias_
        )

        if self.decoder_.bidirectional:
            warnings.warn(
                'The RNN decoder is often unidirectional, you have '
                'initialized a bidirectional decoder, please treat '
                'with caution.'
            )

        return

    def _init_decoder(self):
        r"""An abstract method to initialize decoder encoder.

        Raises
        ------
        NotImplementedError
            When the method is not implemented, an error is raised.
        """

        raise NotImplementedError

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        self.decoder_.reset_parameters()
        self.out_.reset_parameters()

        return


class MLPDecoder(_BasicDecoder):
    """A MLP based decoder.

    Parameters
    ----------
    hidden_dims : int
        The dimension of hidden sample.
    out_dims : int
        The dimension of output sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    sigmoid : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs : dict
        The extra arguments passed to initialize an decoder.

    Attributes
    ----------
    hidden_dims_ : int
        The dimension of hidden sample.
    out_dims_ : int
        The dimension of output sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    sigmoid_ : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs_ : dict
        The extra arguments passed to initialize an decoder.
    decoder_ : torch.nn.Module
        The initialized decoder.
    """

    def _init_decoder(self):
        r"""Initialize a MLP as the decoder.

        Returns
        -------
        decoder : torch.nn.Module
            The MLP based decoder.
        """

        decoder = MLP(
            in_dims=self.hidden_dims_, hidden_dims=self.hidden_dims_,
            out_dims=self.out_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return decoder

    def forward(self, x):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The output samples.
        """

        x = self.decoder_(x)

        if self.sigmoid_:
            x = torch.sigmoid(x)

        return x


class GRUDecoder(_BasicRNNDecoder):
    r"""A GRU based decoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    out_dims : int
        The dimension of output sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    sigmoid : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs : dict
        The extra arguments passed to initialize an decoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    out_dims_ : int
        The dimension of output sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    sigmoid_ : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs_ : dict
        The extra arguments passed to initialize an decoder.
    decoder_ : torch.nn.Module
        The initialized decoder.
    out_ : torch.nn.Module
        The output layer.
    """

    def _init_decoder(self):
        r"""Initialize a GRU as the decoder.

        Returns
        -------
        decoder : torch.nn.Module
            The GRU based decoder.
        """

        decoder = torch.nn.GRU(
            input_size=self.in_dims_, hidden_size=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return decoder

    def forward(self, x, h):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        h : tensor
            The input hidden state.

        Returns
        -------
        tensor
            The output samples.
        tensor
            The output hidden state.
        """

        x, h = self.decoder_(x, h)
        x = self.out_(x)

        if self.sigmoid_:
            x = torch.sigmoid(x)

        return x, h


class LSTMDecoder(_BasicRNNDecoder):
    r"""A LSTM based decoder.

    Parameters
    ----------
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    out_dims : int
        The dimension of output sample.
    num_layers : int
        The number of hidden layers.
    dropout : float
        The dropout rate as a regularization method to each hidden layer.
    bias : bool
        Whether the model will learn an additive bias or not.
    sigmoid : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs : dict
        The extra arguments passed to initialize an decoder.

    Attributes
    ----------
    in_dims_ : int
        The dimension of input sample.
    hidden_dims_ : int
        The dimension of hidden sample.
    out_dims_ : int
        The dimension of output sample.
    num_layers_ : int
        The number of hidden layers.
    dropout_ : float
        The dropout rate as a regularization method to each hidden layer.
    bias_ : bool
        Whether the model will learn an additive bias or not.
    sigmoid_ : bool
        Whether to use sigmoid function over the outputs or not.
    kwargs_ : dict
        The extra arguments passed to initialize an decoder.
    decoder_ : torch.nn.Module
        The initialized decoder.
    out_ : torch.nn.Module
        The output layer.
    """

    def _init_decoder(self):
        r"""Initialize a LSTM as the decoder.

        Returns
        -------
        decoder : torch.nn.Module
            The LSTM based decoder.
        """

        decoder = torch.nn.LSTM(
            input_size=self.in_dims_, hidden_size=self.hidden_dims_,
            num_layers=self.num_layers_,
            dropout=self.dropout_,
            bias=self.bias_,
            **self.kwargs_
        )

        return decoder

    def forward(self, x, h):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The input samples.
        h : tensor
            The input hidden and cell states.

        Returns
        -------
        tensor
            The output samples.
        tensor
            The output hidden and cell states.
        """

        x, h = self.decoder_(x, h)
        x = self.out_(x)

        if self.sigmoid_:
            x = torch.sigmoid(x)

        return x, h
