import torch

from ..functional import kl_with_std_norm


class AE(torch.nn.Module):
    r"""An Auto-Encoder (AE) is a type of artificial
    neural network used to learn efficient data codings
    in an unsupervised manner.

    The aim of an Auto-Encoder is to learn a representation
    for a set of data, typically for dimensionally reduction,
    by training the network to ignore signal "noise".

    The simplest form of an Auto-Encoder is a feedforward,
    non-recurrent neural network similar to single layer
    perceptrons that participate in multilayer perceptrons
    (MLP) - employing an input layer and an output layer
    connected by one or more hidden layers.

    In the simplest case, given one hidden layer, the encoder
    stage of an Auto-Encoder take the input :math:`\mathbf{x}
    \in \mathbb{R}^d = \mathcal{X}` and maps it to
    :math:`\mathbf{h} \in \mathbb{R}^p = \mathcal{F}`:

        .. math::
            \mathbf{h} = \sigma(\mathbf{Wx} + \mathbf{b})

    This image :math:`\mathbf{h}` is usually referred to as
    `code`, `latent variables`, or `latent representation`.

    After that, the decoder stage of an Auto-Encoder maps
    :math:`\mathbf{h}` to the reconstruction :math:`\mathbf{x'}`
    of the same shape as :math:`\mathbf{x}`:

        .. math::
            \mathbf{x'} = \sigma'(\mathbf{W'h} + \mathbf{b'})

    Auto-Encoder are trained to minimize reconstruction errors
    (such as squared errors), often referred to as the loss:

        .. math::
            \mathcal{L}(\mathbf{x}, \mathbf{x'}) =
            \| \mathbf{x} - \mathbf{x'} \|^2

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    decoder : torch.nn.Module
        The decoder used to reconstruct inputs from latent space.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The user defined encoder.
    decoder_ : torch.nn.Module
        The user defined decoder.
    """

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder_ = encoder
        self.decoder_ = decoder

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

        z = self.encoder_(x)
        x = self.decoder_(z)

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

        if hasattr(self.encoder_, 'forward_latent'):
            z = self.encoder_.forward_latent(x)
        else:
            z = self.encoder_(x)

        return z

    def reset_parameters(self):
        r"""Reset the parameters inside.
        """

        self.encoder_.reset_parameters()
        self.decoder_.reset_parameters()

        return


class GAE(AE):
    r"""A GNN based Auto-Encoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    decoder : torch.nn.Module
        The decoder used to reconstruct inputs from latent space.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The user defined encoder.
    decoder_ : torch.nn.Module
        The user defined decoder.
    """

    def forward(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The node features of the graph.
        edge_index : tensor
            The adjacency matrix of the graph.

        Returns
        -------
        tensor
            The output samples.
        """

        z = self.encoder_(x, edge_index, *args, **kwargs)
        x = self.decoder_(z)

        return x

    def forward_latent(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The node features of the graph.
        edge_index : tensor
            The adjacency matrix of the graph.

        Returns
        -------
        tensor
            The latent samples.
        """

        if hasattr(self.encoder_, 'forward_latent'):
            z = self.encoder_.forward_latent(x, edge_index, *args, **kwargs)
        else:
            z = self.encoder_(x, edge_index, *args, **kwargs)

        return z


class RNNAE(AE):
    r"""A RNN based Auto-Encoder.

    For RNN Auto-Encoder, the encoder and decoder inside a member
    of RNN based Neural Network.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    decoder : torch.nn.Module
        The decoder used to reconstruct inputs from latent space.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The user defined encoder.
    decoder_ : torch.nn.Module
        The user defined decoder.
    """

    def forward(self, x):
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
        """

        h = self.encoder_(x)
        x, _ = self.decoder_(x, h)

        return x


class VAE(AE):
    r"""Variational Auto-Encoder (VAE) is generative model, akin to
    generative adversarial networks. VAEs are directed probabilistic
    graphical models (DPGM) whose posterior is approximated by a
    neural network, forming an Auto-Encoder like architecture.

    The encoder model in this setup can be inherited from any basic
    Auto-Encoder encoder architectures. In particular, we use two
    separate Linear transformations to generate mean and variance
    parameters, respectively, condition on this input:

        .. math::
            \mu_{\mathbf{Z}} =
            \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}) \\
            \log \sigma_{\mathbf{Z}} =
            \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})

    Here, :math:`\mu_{\mathbf{Z}}` is mean value for latent embedding
    variables. The :math:`\log \sigma_{\mathbf{Z}}` specifies the log-
    variance for the latent embedding.

    Given the encoded :math:`\mu_{\mathbf{Z}}` and :math:`\log \sigma_
    {\mathbf{Z}}` parameters, we can sample a set of latent embedding
    variables by computing

        .. math::
            \mathbf{Z} = \epsilon \circ \exp (\log(\sigma_{\mathbf{Z}})) +
            \mu_{\mathbf{Z}}

    where :math:`\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{1})` is
    independently sampled unit normal entries.

    After that, the decoder stage of an Variational Auto-Encoder maps
    :math:`\mathbf{Z}` to the reconstruction :math:`\mathbf{x'}`
    of the same shape as :math:`\mathbf{x}`:

        .. math::
            \mathbf{x'} = \sigma'(\mathbf{W'Z} + \mathbf{b'})

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    decoder : torch.nn.Module
        The decoder used to reconstruct inputs from latent space.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The user defined encoder.
    decoder_ : torch.nn.Module
        The user defined decoder.
    """

    mu_, logstd_ = torch.zeros(1, 1), torch.zeros(1, 1)

    def reparametrize(self, mu, logstd):
        r"""Sampling a set of latent embeddings.

        Parameters
        ----------
        mu : tensor
            The mean result of encoder.
        logstd : tensor
            The log variance result of encoder.

        Returns
        -------
        tensor
            The sampling result of latent node embeddings.
        """

        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

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

        self.mu_, self.logstd_ = self.encoder_(x)
        z = self.reparametrize(self.mu_, self.logstd_)
        x = self.decoder_(z)

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

        self.mu_, self.logstd_ = self.encoder_(x)
        z = self.reparametrize(self.mu_, self.logstd_)

        return z

    def kl_loss(self):
        r"""Compute KL loss of current VAE.

        It will use the latest obtained mean and log standard
        deviation of samples in the forward pass to compute
        current KL loss.

        Returns
        -------
        loss : tensor
            The KL loss to a standard normal distribution.
        """

        kl_loss = kl_with_std_norm(
            self.mu_, self.logstd_,
            reduction='batchmean'
        )

        return kl_loss


class VGAE(VAE):
    r"""A GNN based Variational Auto-encoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    decoder : torch.nn.Module
        The decoder used to reconstruct inputs from latent space.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The user defined encoder.
    decoder_ : torch.nn.Module
        The user defined decoder.
    """

    def forward(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        x : tensor
            The node features of the graph.
        edge_index : tensor
            The adjacency matrix of the graph.

        Returns
        -------
        tensor
            The output samples.
        """

        self.mu_, self.logstd_ = self.encoder_(x, edge_index, *args, **kwargs)
        z = self.reparametrize(self.mu_, self.logstd_)
        x = self.decoder_(z)

        return x

    def forward_latent(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain latent samples.

        Parameters
        ----------
        x : tensor
            The node features of the graph.
        edge_index : tensor
            The adjacency matrix of the graph.

        Returns
        -------
        tensor
            The latent samples.
        """

        self.mu_, self.logstd_ = self.encoder_(x, edge_index, *args, **kwargs)
        z = self.reparametrize(self.mu_, self.logstd_)

        return z
