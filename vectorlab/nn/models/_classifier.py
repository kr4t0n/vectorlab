import torch


class Classifier(torch.nn.Module):
    r"""A basic classifier.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    classifier : torch.nn.Module
        The classifier to classify inputs using their latent
        representations.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The encoder used to encode inputs to latent space.
    classifier_ : torch.nn.Module
        The classifier to classify inputs using their latent
        representations.
    """

    def __init__(self, encoder, classifier):

        super().__init__()

        self.encoder_ = encoder
        self.classifier_ = classifier

        return

    def forward(self, x):
        r"""The forward process to obtain the classified result.

        Parameters
        ----------
        x : tensor
            The input samples.

        Returns
        -------
        tensor
            The output classified result.
        """

        if hasattr(self.encoder_, 'forward_latent'):
            z = self.encoder_.forward_latent(x)
        else:
            z = self.encoder_(x)

        y_hat = self.classifier_(z)

        return y_hat

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
        self.classifier_.reset_parameters()

        return


class GClassifier(Classifier):
    r"""A base GNN classifier.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder used to encode inputs to latent space.
    classifier : torch.nn.Module
        The classifier to classify inputs using their latent
        representations.

    Attributes
    ----------
    encoder_ : torch.nn.Module
        The encoder used to encode inputs to latent space.
    classifier_ : torch.nn.Module
        The classifier to classify inputs using their latent
        representations.
    """

    def forward(self, x, edge_index, *args, **kwargs):
        r"""The forward process to obtain the classified result.

        Parameters
        ----------
        x : tensor
            The node features of the graph.
        edge_index : tensor
            The adjacency matrix of the graph.

        Returns
        -------
        tensor
            The output classified result.
        """

        if hasattr(self.encoder_, 'forward_latent'):
            z = self.encoder_.forward_latent(x, edge_index, *args, **kwargs)
        else:
            z = self.encoder_(x, edge_index, *args, **kwargs)

        y_hat = self.classifier_(z)

        return y_hat

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
