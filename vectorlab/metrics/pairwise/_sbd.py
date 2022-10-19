"""
SBD stands for shape based distance, which is a
measurement of how far two time series are.
The measurement is focused on the shape of time series.
"""

import numpy as np

from sklearn.metrics.pairwise import check_pairwise_arrays


def _shape_based_distance_shifting(Y, s):
    r"""Shift the time series, with desired step `s`, the shifted
    part is filled with zeros.

    Parameters
    ----------
    Y : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input time series.
    s : int
        Shifted steps.

    Returns
    -------
    {array-like, sparse matrix}, shape (n_samples, n_features)
        Shifted time series.
    """

    if s > 0:
        return np.concatenate(
            [
                np.zeros((s, Y.shape[-1])),
                Y[:-s]
            ]
        )
    else:
        if -s == len(Y):
            return np.zeros((-s, Y.shape[-1]))
        else:
            return np.concatenate(
                [
                    Y[-s - len(Y):],
                    np.zeros((-s, Y.shape[-1]))
                ]
            )


def sbd(X, Y=None, shifting=True):
    r"""Cross-correlation calculates the sliding inner-product
    of two time series, which is natively robust to phase shift.
    Shape-based distance (SBD) on the basis of cross-correlation
    and applied it on idealized time series data.

    For two time series,

        .. math::
            X = [x_1, x_2, x_3, \ldots, x_n] \\
            Y = [y_1, y_2, y_3, \ldots, y_n]

    we calculate the inner-product CCs(X, Y) as the similarity
    between time series X and Y with phase shift s as,

        .. math::
            CCs(X, Y) =\begin{cases}
                \sum_{i=1}^{m-s} x_{s+i} * y_{i}, & s \geq 0\\
                \sum_{i=1}^{m+s} x_{i} * y_{i-s}, & s \lt 0 \end{cases}

    The cross-correlation is the maximized value of CCs(X, Y) as,

        .. math::
            \begin{align*}NCC(X, Y) &=
                \max_{s} CCs(X, Y) / (\|X\|_{2}* \|Y\|_{2})\\
                SBD(X, Y) &= 1 - NCC(X, Y) \end{align*}

    SBD ranges from 0 to 2, where 0 means two time series have exactly the
    same shape. A smaller SBD means higher shape similarity.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        The second input data.
    shifting : bool, optional
        Enable shifting to calculate SBD.

    Returns
    -------
    distance : float
        The distance between X and Y.
    """

    X, Y = check_pairwise_arrays(X, Y)

    ccs_value = float('-inf')
    if shifting:
        for s in range(-len(X) + 1, len(X)):
            value = np.tensordot(
                X,
                _shape_based_distance_shifting(Y, s)
            )
            if value > ccs_value:
                ccs_value = value
    else:
        ccs_value = np.tensordot(X, Y)

    ncc_value = ccs_value / \
        (np.linalg.norm(X, ord=2) * np.linalg.norm(Y, ord=2))

    # Clipping ncc_value, since there involves sqrt operation
    ncc_value = np.clip(ncc_value, -1, 1)

    distance = 1 - ncc_value

    return distance
