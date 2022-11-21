import torch

from ._mlp import MLP


class BasicDecoder(torch.nn.Module):
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
    kwargs_ : dict
        The extra arguments passed to initialize an decoder.
    decoder_ : torch.nn.Module
        The initialized decoder.
    """

    def __init__(self, hidden_dims, out_dims, num_layers,
                 dropout=.5,
                 bias=True,
                 **kwargs):

        super().__init__()

        self.hidden_dims_ = hidden_dims
        self.out_dims_ = out_dims
        self.num_layers_ = num_layers
        self.dropout_ = dropout
        self.bias_ = bias
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


class MLPDecoder(BasicDecoder):
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

        return x
