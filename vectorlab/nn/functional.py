import warnings


def apply_loss_reduction(loss, reduction):
    r"""Apply the reduction method to the computed loss.

    The reduction method could be one of the 'none', 'mean' and
    'sum'. If the reduction method is 'none', it will directly
    return the loss value in batch size, otherwise a scalar will
    be returned. The 'mean' and 'sum' operation will be operated
    over batch size and support size.

    Parameters
    ----------
    loss : tensor, shape (n_batches)
        The batch of loss values.
    reduction : str
        Specifies the reduction to apply to the output.

    Returns
    -------
    loss : tensor
        The reduced loss value.
    """

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()


def kl_with_std_norm(mu, logstd, reduction='mean'):
    r"""The kl divergence to a standard normal distribution.

    Parameters
    ----------
    mu : tensor
        The mean of samples.
    logstd : tensor
        The log standard deviation of samples.
    reduction : str, optional
        Specifies the reduction to apply to the output.

    Returns
    -------
    tensor
        The result of kl divergence to a standard normal distribution.
    """

    reduced = -0.5 * (1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2)

    if reduction == 'mean':
        warnings.warn(
            'reduction: mean divides the total loss by both the batch size '
            'and the support size. However, batchmean divides only by the '
            'batch size, and aligns with the KL math definition. As a result '
            'mean reduction in KL will be same as the batchmean reduction.'
        )
        reduction = 'batchmean'

    if reduction == 'batchmean':
        reduced = apply_loss_reduction(reduced, reduction='sum')
        reduced = reduced / mu.size()[0]
    else:
        reduced = apply_loss_reduction(reduced, reduction=reduction)

    return reduced
