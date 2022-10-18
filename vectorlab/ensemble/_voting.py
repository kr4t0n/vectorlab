"""
Voting strategies for ensemble learning.
"""

import numpy as np

from ..utils._check import check_valid_option


def _soft_voting(results, weights=None):
    r"""Given the various results from different sources,
    use soft voting method to conclude the final result.

    Each result should have the same dimension, in theory,
    each result has the shape of number of samples and number
    of possibilities. Soft voting will average all the
    possibilities from different sources, and find the most
    possible result. If weights are provided, soft voting
    will perform in a weighted average manner.

    Parameters
    ----------
    results : tuple
        The tuple of results from different sources.
    weights : list, np.ndarray, optional
        The importance of result from different sources.

    Returns
    -------
    _voting : np.ndarray
        The soft voting result.
    """

    assert np.all(results[0].shape == result.shape for result in results)

    if weights is not None:
        assert len(results) == len(weights)

    results = np.stack(results)
    soft_results = np.average(results, axis=0, weights=weights)

    _voting = np.argmax(soft_results, axis=1)

    return _voting


def _hard_voting(results, weights=None):
    r"""Given the various results from different sources,
    use hard voting method to conclude the final result.

    Each result should have the same dimension, in theory,
    each result has the shape of number of samples. Hard
    voting will select the results from different sources
    with the most occurrences. If weights are provided, hard
    voting will perform in a weighted sum manner.

    Parameters
    ----------
    results : tuple
        The tuple of results from different sources.
    weights : list, np.ndarray, optional
        The importance of result from different sources.

    Returns
    -------
    _voting : np.ndarray
        The hard voting result.
    """

    assert np.all(results[0].shape == result.shape for result in results)

    if weights is not None:
        assert len(results) == len(weights)

    results = np.stack(results)

    _voting = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x, weights=weights)),
        axis=0,
        arr=results
    )

    return _voting


def voting(results, weights=None, method='soft'):
    r"""Given the various results from different sources,
    use voting method to conclude the final result.

    Each result should have the same dimension, in theory,
    each result has the shape of number of samples and number
    of possibilities. Voting could choose different methods,
    soft or hard way. When using soft voting, the possibilities
    from different sources will be averaged, and then find the
    most possible result, while the hard voting will first find
    the most possible result in each sources and then select the
    result with the most occurrences.

    Parameters
    ----------
    results : tuple
        The tuple of results from different sources.
    weights : list, np.ndarray, optional
        The importance of result from different sources.
    method : str, optional
        The method used to generate the voting result.

    Returns
    -------
    _voting : np.ndarray
        The voting result.
    """

    method = check_valid_option(
        method, ['soft', 'hard'],
        variable_name='voting method'
    )

    if method == 'soft':
        _voting = _soft_voting(results, weights=weights)
    elif method == 'hard':
        results = tuple(np.argmax(result, axis=1) for result in results)
        _voting = _hard_voting(results, weights=weights)

    return _voting
