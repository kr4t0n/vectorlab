"""
This module provides basic utilities operated on array data.
"""

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.optimize import quadratic_assignment

from ..utils._check import _check_ndarray, check_valid_option
from ..optimize._qap import (
    _quadratic_assignment_lawler, _quadratic_assignment_yaaq
)


def replace_array(origin_arr, origin_values, replace_values):
    r"""Replace the values inside array with new values.

    This function will replace values inside the array with
    replacements.

    Parameters
    ----------
    origin_arr : array_like
        The original array to be replaced.
    origin_values : array_like, shape (n_replacements)
        The original values to be replaced inside the array.
    replace_values : array_like, shape (n_replacements)
        The replacements values to replace the original values.

    Returns
    -------
    arr : array_like, shape (n_samples)
        The replaced array.
    """

    origin_arr = _check_ndarray(origin_arr)
    origin_values = _check_ndarray(origin_values)
    replace_values = _check_ndarray(replace_values)

    arr = origin_arr.copy()

    arr[
        tuple(
            map(
                np.concatenate,
                zip(
                    *[
                        np.where(arr == origin_value)
                        for origin_value in origin_values
                    ]
                )
            )
        )
    ] = np.concatenate(
        [
            np.tile(replace_value, np.where(arr == origin_value)[0].shape)
            for origin_value, replace_value
            in zip(origin_values, replace_values)
        ]
    )

    return arr


def split_array(origin_arr, interval):
    r"""Split the values inside array with a certain interval.

    This function will splice values inside the array with a certain
    interval. For any difference between two adjacent values inside
    the split part, the difference is lower than the interval, while
    the difference between two split parts are higher than the interval.

    Parameters
    ----------
    origin_arr : array_like, shape (n_samples)
        The original array to be split.
    interval : int
        The interval used to split the array.

    Returns
    -------
    arr_split : list
        The list of split parts, each part is array_like.
    """

    origin_arr = _check_ndarray(origin_arr)

    arr = np.sort(origin_arr)

    arr_diff = np.diff(arr)
    arr_split = np.split(arr, (np.where(arr_diff > interval)[0] + 1))

    return arr_split


def swap_matrix(origin_mat, transform=None, return_transform=False):
    r"""Swap the matrix with a new order of indices.

    This function will transform a matrix using a new order of indices,
    while the new order will not only swap between rows but also swap
    between columns.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix to be swapped.
    transform : array_like, shape (n_samples), optional
        The transform used to swap the original matrix. If it is not
        provided, the transform will generate randomly.
    return_transform : bool, optional
        If return the transformation being performed.

    Returns
    -------
    transformed_mat : array_like, shape (n_samples, n_samples)
        The transformed matrix swapped using new order of indices.
    """

    origin_mat = _check_ndarray(origin_mat)

    if transform is None:
        transform = np.arange(0, origin_mat.shape[0])
        np.random.shuffle(transform)

    transformed_mat = \
        origin_mat.take(transform, axis=0).take(transform, axis=1)

    if return_transform:
        return (transformed_mat, transform)
    else:
        return transformed_mat


def _find_forward_transform_unary(origin_mat, transformed_mat, diff_mat=None):
    r"""Using an unary similarity matrix to find the proper transformation.

    This function is not guaranteed to find the right transformation. If used
    a node-wise difference matrix, an unary similarity matrix to approximately
    calculate the cost to assign a node pair. It is preferred that difference
    matrix provided, as the quality of such difference matrix matters the
    quality of assignments. if the difference matrix is not provided, this
    function will simply calculate the degree difference between each pair of
    nodes as the difference matrix.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix.
    transformed_mat : array_like, shape (n_samples, n_samples)
        THe transformed matrix.
    diff_mat : None, array_like, shape (n_samples, n_samples)
        The unary difference matrix.

    Returns
    -------
    transform : array_like, shape (n_samples)
        A possible transform.
    """

    if diff_mat is None:

        diff_mat = np.empty((transformed_mat.shape[0], origin_mat.shape[0]))

        for i in range(transformed_mat.shape[0]):
            for j in range(origin_mat.shape[0]):

                diff_mat[i][j] = np.abs(
                    np.sum(transformed_mat[i]) - np.sum(origin_mat[j])
                )

    transform = linear_sum_assignment(diff_mat)[1]

    return transform


def _find_forward_transform_pairwise(origin_mat, transformed_mat):
    r"""Using a pairwise similarity matrix to find the proper transformation.

    This function is not guaranteed to find the right transformation.
    If used a edge-wise difference matrix, a pairwise similarity matrix
    to approximately calculate the cost to assign a node pair. The
    origin_mat and transformed_mat are used to calculate the edge difference
    once a pair of nodes are matched. The origin_mat and transformed_mat could
    be either in a discrete manner containing zero and one or in a continuous
    manner representing the edge weight.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix.
    transformed_mat : array_like, shape (n_samples, n_samples)
        The transformed matrix.

    Returns
    -------
    transformed : array_like, shape (n_samples)
        A possible transform.
    """

    res = quadratic_assignment(
        transformed_mat, origin_mat,
        options={'maximize': True}
    )

    return res['col_ind']


def _find_forward_transform_lawler(origin_mat, transformed_mat, aggr='sum'):
    r"""Combine unary and pairwise similarity matrix to find the proper
    transformation.

    This function is not guaranteed to find the right transformation.
    It combines the unary and pairwise similarity into a combined form,
    lawler form to calculate the cost to assign a node pair. The origin_mat
    and transformed_mat are used to calculate the node difference and edge
    difference once a pair of nodes are matched. The origin_mat and
    transformed_mat could be either in a discrete manner containing zero and
    one or in a continuous manner representing the edge weight.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix.
    transformed_mat : array_like, shape (n_samples, n_samples)
        The transformed matrix.
    aggr : str, optional
        The aggregation method used to compute the gradient.

    Returns
    -------
    transformed : array_like, shape (n_samples)
        A possible transform.
    """

    res = _quadratic_assignment_lawler(
        transformed_mat, origin_mat,
        maximize=True,
        aggr=aggr
    )

    return res['col_ind']


def _find_forward_transform_yaaq(origin_mat, transformed_mat):
    r"""Combine unary and pairwise similarity matrix to find the proper
    transformation.

    This function is not guaranteed to find the right transformation.
    It combines the unary and pairwise similarity into a combined form,
    lawler form to calculate the cost to assign a node pair. The origin_mat
    and transformed_mat are used to calculate the node difference and edge
    difference once a pair of nodes are matched. The origin_mat and
    transformed_mat could be either in a discrete manner containing zero and
    one or in a continuous manner representing the edge weight.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix.
    transformed_mat : array_like, shape (n_samples, n_samples)
        The transformed matrix.
    aggr : str, optional
        The aggregation method used to compute the gradient.

    Returns
    -------
    transformed : array_like, shape (n_samples)
        A possible transform.
    """

    res = _quadratic_assignment_yaaq(
        transformed_mat, origin_mat,
        maximize=True
    )

    return res['col_ind']


def find_forward_transform(origin_mat, transformed_mat,
                           method='yaaq', **kwargs):
    r"""Find a transform that could be used to transform original matrix
    to transformed matrix.

    This function is not guaranteed to find the right transformation.
    There are two methods to find such transformation, using an unary
    similarity matrix or a pairwise similarity matrix. When using an
    unary matrix, it could be seen as an assignment problem for linear
    programming, while using a pairwise matrix, it could be seen as an
    assignment problem for quadratic programming.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix.
    transformed_mat : array_like, shape (n_samples, n_samples)
        The transformed matrix.
    method : str, optional
        The method used to solve this problem, could be one of unary
        pairwise and combined.
    **kwargs : dict
        Extra arguments used in particular solving method.

    Returns
    -------
    transform : array_like (n_samples)
        A possible transform.
    """

    origin_mat = _check_ndarray(origin_mat)
    transformed_mat = _check_ndarray(transformed_mat)

    method = check_valid_option(
        method, ['unary', 'pairwise', 'lawler', 'yaaq'],
        variable_name='finding assignment method'
    )

    if method == 'unary':
        transform = _find_forward_transform_unary(
            origin_mat, transformed_mat,
            **kwargs
        )

    if method == 'pairwise':
        transform = _find_forward_transform_pairwise(
            origin_mat, transformed_mat,
            **kwargs
        )

    if method == 'lawler':
        transform = _find_forward_transform_lawler(
            origin_mat, transformed_mat,
            **kwargs
        )

    if method == 'yaaq':
        transform = _find_forward_transform_yaaq(
            origin_mat, transformed_mat,
            **kwargs
        )

    return transform


def find_backward_transform(origin_mat, transformed_mat,
                            method='yaaq', **kwargs):
    r"""Find a transform that could be used to transform transformed matrix
    back to original matrix.origin_mat

    This function is not guaranteed to find the right transform. Since finding
    the transform that could perform backward operation is just simply the
    reverse way to find the forward one. We simply use the forward transformed
    with opposite arguments. Detailed information could be find in the forward
    function.

    Parameters
    ----------
    origin_mat : array_like, shape (n_samples, n_samples)
        The original matrix.
    transformed_mat : array_like, shape (n_samples, n_samples)
        The transformed matrix.

    Returns
    -------
    transform : array_like, shape (n_samples)
        A possible transform.
    """

    origin_mat = _check_ndarray(origin_mat)
    transformed_mat = _check_ndarray(transformed_mat)

    transform = find_forward_transform(
        transformed_mat, origin_mat,
        method=method, **kwargs
    )

    return transform
