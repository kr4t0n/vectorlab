import torch
import torch.nn.functional as F

from ...utils._check import check_valid_int
from .._resolver import activation_resolver, nn_normalization_resolver


class MLP(torch.nn.Module):
    r"""A Multi-layer Perception model.

    Multi-layer Perception model is a model stacked with multiple linear
    layers, with non-linearity and normalization between linear layers.

    Parameters
    ----------
    dim_list : list
        The list of different dimensions of each linear layer.
    in_dims : int
        The dimension of input sample.
    hidden_dims : int
        The dimension of hidden sample.
    out_dims : int
        The dimension of output sample.
    num_layers : int
        The number of linear layers.
    act_fn : str, callable
        The non-linear activation function to use.
    act_first : bool
        Whether the activation function is used before normalization or not.
    act_kwargs : dict
        The extra arguments passed to activation function.
    norm_fn : str, callable
        The normalization layer to use.
    norm_kwargs : dict
        The extra arguments passed to normalization layer.
    dropout : float, list
        The dropout rate as a regularization method to each hidden layer.
    plain_last : bool
        Whether the non-linearity and normalization is not applied to
        the last layer or not.
    bias : bool, list
        Whether the model will learn an additive bias or not.

    Attributes
    ----------
    dim_list_ : list
        The list of different dimension of each linear layer.
    act_fn_ : callable
        The resolved non-linear acitvation function to use.
    act_first_ : bool
        Whether the activation function is used before normalization or not.
    act_kwargs_ : dict
        The extra arguments passed to activation function.
    act_ : callable
        The actual activation function to use.
    norm_fn_ : callable
        The resolved normalization layer to use.
    norm_kwargs_ : dict
        The extra arguments passed to normalization layer.
    dropout_ : list
        The dropout rate as a regularization method to each hidden layer.
    plain_last_ : bool
        Whether the non-linearity and normalization is not applied to
        the last layer or not.
    bias_ : list
        Whether the model will learn an additive bias or not to each
        hidden layer.
    lins_ : torch.nn.ModuleList
        The list of all linear layers.
    norms_ : torch.nn.ModuleList
        The list of all normalization layers.
    """

    def __init__(self,
                 dim_list=None,
                 in_dims=None, hidden_dims=None, out_dims=None,
                 num_layers=None,
                 act_fn='relu',
                 act_first=False, act_kwargs=None,
                 norm_fn='batch_norm_1d',
                 norm_kwargs=None,
                 dropout=.0,
                 plain_last=True,
                 bias=True):

        super().__init__()

        if isinstance(dim_list, int):
            in_dims = dim_list

        if (
            in_dims is not None
        ) and (
            hidden_dims is not None
        ) and (
            out_dims is not None
        ):

            num_layers = check_valid_int(
                num_layers,
                lower=1,
                variable_name='number of layers'
            )

            dim_list = [hidden_dims] * (num_layers - 1)
            dim_list = [in_dims] + dim_list + [out_dims]

        assert isinstance(dim_list, (tuple, list)), (
            'Failed to construct proper dimension list to'
            'initialize MLP.'
        )
        self.dim_list_ = dim_list

        self.act_fn_ = act_fn
        self.act_first_ = act_first
        self.act_kwargs_ = act_kwargs if act_kwargs else {}
        # Resolve activation function
        self.act_fn_ = activation_resolver(self.act_fn_)
        if self.act_fn_ is None:
            self.act_ = None
        else:
            self.act_ = self.act_fn_(**self.act_kwargs_)

        self.norm_fn_ = norm_fn
        self.norm_kwargs_ = norm_kwargs if norm_kwargs else {}
        # Resolve batch normalization function
        self.norm_fn_ = nn_normalization_resolver(self.norm_fn_)

        self.plain_last_ = plain_last
        if isinstance(dropout, float):
            dropout = [dropout] * (len(self.dim_list_) - 1)
            if self.plain_last_:
                dropout[-1] = .0
        if len(dropout) != len(self.dim_list_) - 1:
            raise ValueError(
                f'Number of dropout values provided {len(dropout)} '
                f'does not match the number of layers specified '
                f'{len(self.dim_list_) - 1}'
            )
        self.dropout_ = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(self.dim_list_) - 1)
        if len(bias) != len(self.dim_list_) - 1:
            raise ValueError(
                f'Number of bias values provided {len(bias)} '
                f'does not match the number of layers specified '
                f'{len(self.dim_list_) - 1}'
            )
        self.bias_ = bias

        self.lins_ = torch.nn.ModuleList()
        iterator = zip(self.dim_list_[:-1], self.dim_list_[1:], bias)
        for in_dims_, out_dims_, bias_ in iterator:
            self.lins_.append(
                torch.nn.Linear(in_dims_, out_dims_, bias=bias_)
            )

        self.norms_ = torch.nn.ModuleList()
        iterator = \
            self.dim_list_[1:-1] if self.plain_last_ else self.dim_list_[1:]
        for hidden_dims_ in iterator:
            if self.norm_fn_ is None:
                self.norms_.append(
                    torch.nn.Identity()
                )
            else:
                self.norms_.append(
                    self.norm_fn_(hidden_dims_, **self.norm_kwargs_)
                )

        return

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

        for i, (lin, norm) in enumerate(zip(self.lins_, self.norms_)):
            x = lin(x)

            if self.act_ and self.act_first_:
                x = self.act_(x)

            x = norm(x)

            if self.act_ and not self.act_first_:
                x = self.act_(x)

            x = F.dropout(x, p=self.dropout_[i], training=self.training)

        if self.plain_last_:
            x = self.lins_[-1](x)
            x = F.dropout(x, p=self.dropout_[-1], training=self.training)

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

        for i, (lin, norm) in enumerate(zip(self.lins_, self.norms_)):
            x = lin(x)

            if self.act_ and self.act_first_:
                x = self.act_(x)

            x = norm(x)

            if self.act_ and not self.act_first_:
                x = self.act_(x)

            x = F.dropout(x, p=self.dropout_[i], training=self.training)

        return x

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        for lin in self.lins_:
            lin.reset_parameters()
        for norm in self.norms_:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

        return
