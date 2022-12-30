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
        reduced = reduced / mu.shape[0]
    else:
        reduced = apply_loss_reduction(reduced, reduction=reduction)

    return reduced


def graph_recon_loss(adj, pos_edge_index, neg_edge_index, reduction='mean'):
    r"""Compute graph reconstruction loss.

    Parameters
    ----------
    adj : tensor
        The adjacency matrix of reconstructed graph.
    pos_edge_index : tensor
        The positive edge index.
    neg_edge_index : tensor
        The negative edge index.
    reduction : str, optional
        Specifies the reduction to apply to the output.

    Returns
    -------
    tensor
        The resulf computed graph reconstruction loss.
    """

    eps = 1e-16

    pos_loss = - (adj[pos_edge_index[0], pos_edge_index[1]] + eps).log()
    pos_reduced = apply_loss_reduction(pos_loss, reduction=reduction)

    neg_loss = - ((1 - adj[neg_edge_index[0], neg_edge_index[1]]) + eps).log()
    neg_reduced = apply_loss_reduction(neg_loss, reduction=reduction)

    reduced = pos_reduced + neg_reduced

    return reduced
