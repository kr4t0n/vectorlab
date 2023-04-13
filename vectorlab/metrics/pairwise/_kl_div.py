"""
Kullback-Leibler divergence, also called relative
entropy, is a measure of how one probability distribution
is different from a second.
"""

import numpy as np

from scipy import stats, integrate

from ...utils._check import check_nd_array, check_pairwise_1d_array


def kl_div(X, Y=None,
           weights_X=None, weights_Y=None,
           eps=1e-40):
    r"""In mathematical statistics, the Kullback-Leibler divergence,
    :math:`D_{KL}`, also called relative entropy, is a measure of how
    one probability distribution is different from a second.

    For two time series data, X and Y, we first compute their density
    function using Gaussian kernel density estimator. The kernel density
    estimator is

        .. math::
            \hat{f_h}(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i) =
            \frac{1}{nh} \sum_{i=1}^{n} K(\frac{x - x_i}{h})

    where :math:`K` is the kernel - a non-negative function - and
    :math:`h \gt 0` is a smoothing parameter called the `bandwidth`.
    As the form of Gaussian kernel

        .. math::
            K_h(x) \propto \exp(- \frac{x^2}{2h^2})

    When two kernel density functions are computed, we re-sample from
    these two functions, and compute the KL divergence between them.

    For discrete probability distributions :math:`P` and :math:`Q` defined
    on the same probability space, :math:`\mathcal X`, the relative entropy
    from :math:`Q` to :math:`P` is defined to be

        .. math::
            D_{KL}(P \| Q) = \sum_{x \in \mathcal X} P(x)
            \log(\frac{P(x)}{Q(x)})

    which is equivalent to

        .. math::
            D_{KL}(P \| Q) = - \sum_{x \in \mathcal X} P(x)
            \log(\frac{Q(x)}{P(x)})

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples), optional
        The second input data.
    weights_X : {array-like, sparse matrix}, shape (n_samples), optional
        The weights of the first input data.
    weights_Y : {array-like, sparse matrix}, shape (n_samples), optional
        The weights of the second input data.
    eps : float, optional
        The minimal eps to avoid runtime overflow.

    Returns
    -------
    divergence : float
        The divergence between X and Y.
    """

    X = check_nd_array(X, n=1)
    if Y is None:
        Y = X
    else:
        Y = check_nd_array(Y, n=1)

    if weights_X is not None:
        X, weights_X = check_pairwise_1d_array(X, weights_X)

    if weights_Y is not None:
        Y, weights_Y = check_pairwise_1d_array(Y, weights_Y)

    kde_X = stats.gaussian_kde(X, weights=weights_X)
    kde_Y = stats.gaussian_kde(Y, weights=weights_Y)

    X_mu = integrate.quad(
        lambda x: x * kde_X.pdf(x),
        a=-np.Inf, b=np.Inf
    )[0]
    X_var = integrate.quad(
        lambda x: (x ** 2) * kde_X.pdf(x),
        a=-np.inf, b=np.inf
    )[0] - X_mu ** 2

    Y_mu = integrate.quad(
        lambda y: y * kde_Y.pdf(y),
        a=-np.Inf, b=np.Inf
    )[0]
    Y_var = integrate.quad(
        lambda y: (y ** 2) * kde_Y.pdf(y),
        a=-np.inf, b=np.inf
    )[0] - Y_mu ** 2

    var_ratio = X_var / Y_var
    t1 = ((X_mu - Y_mu) ** 2) / Y_var

    divergence = 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))

    return divergence
