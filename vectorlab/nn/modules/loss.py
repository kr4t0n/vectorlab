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
        r"""The forward process to obtain loss result.

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

        loss = F.kl_with_std_norm(
            mu, logstd,
            reduction=self.reduction
        )

        return loss


class SequenceNLLLoss(torch.nn.NLLLoss):
    r"""The NLL loss for sequence data.

    Parameters
    ----------
    weight : tensor, optional
        A manual rescaling weight given to each class.
    reduction : str, optional
        Specifies the reduction to apply to the output.
    batch_first : bool, optional
        If true, the first dimension is batch size, if false,
        the second dimension is batch size.
    """

    def __init__(self, weight=None, reduction='mean', batch_first=False):

        super().__init__(weight=weight, reduction=reduction)

        self.batch_first = batch_first

        return

    def forward(self, input, target):
        r"""The forward process to obtain loss result.

        Parameters
        ----------
        input : tensor
            The input probabilities for all classes.
        target : tensor
            The target label for all classes.

        Returns
        -------
        loss : tensor
            The NLL loss for sequence data.
        """

        if self.batch_first:
            # input shape should be (batch_size, , n_classes)
            # permute to ==> (batch_size, n_classes, n_seqs)
            input = input.permute(0, 2, 1)
        else:
            # input shape should be (n_seqs, batch_size, n_classes)
            # permute to ==> (batch_size, n_classes, n_seqs)
            # input shape should be (n_seqs, batch_size)
            # permute to ==> (batch_size, n_seqs)
            input = input.permute(1, 2, 0)
            target = target.permute(1, 0)

        return super().forward(input, target)
