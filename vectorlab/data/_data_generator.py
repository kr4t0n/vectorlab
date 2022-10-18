"""
Stochastic data generators.
"""

import numpy as np

from ..utils._check import check_nd_array, check_valid_int


def data_linear_generator(n_samples,
                          w, b=0,
                          noise=True, eps=1e-2):
    r"""Generate linear data.

    Generate X based on the length of w, weights. The shape of X
    would be (n_samples, len(w)). Generate y according to the
    parameters, w and b. In short, :math:`y = Xw + b`. Noise will
    be added according to the parameter noise.

    Parameters
    ----------
    n_samples : int
        The number of samples to be generated.
    w : array_like, (n_dims, )
        Weights.
    b : int, array_like, (n_dims, ), optional
        Bias.
    noise : bool, optional
        If add noise to target y.
    eps : float, optional
        The range of noise to add.

    Returns
    -------
    X, array_like, (n_samples, n_dims)
        X generated.
    y, array_like, (n_samples)
        y generated.
    """

    n_samples = check_valid_int(
        n_samples,
        lower=1,
        variable_name='n_samples'
    )
    w = check_nd_array(w, n=1)

    n_dims = w.shape[0]

    X = np.random.normal(0, 1, (n_samples, n_dims))
    y = np.matmul(X, w) + b

    if noise:
        y += np.random.normal(0, eps, y.shape)

    return X, y
