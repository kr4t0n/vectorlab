import torch

from .. import functional as F


class KLWithStdNormLoss(torch.nn.modules.loss._Loss):
    r"""The Kullback-Leibler divergence loss to a standard normal
    distribution.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output.
    """

    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):

        super().__init__(reduction=reduction)

        return

    def forward(self, mu, logstd):
        r"""The forward process to obtain output samples.

        Parameters
        ----------
        mu : tensor
            The mean of samples.
        logstd : tensor
            The log standard deviation of samples.

        Returns
        -------
        loss : tensor
            The KL loss to a standard normal distribution.
        """

        loss = F.kl_with_std_norm(mu, logstd, self.reduction)

        return loss
