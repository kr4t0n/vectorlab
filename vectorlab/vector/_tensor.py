"""
This module provides basic utilities operated on tensor.
"""

import torch

from ..utils._check import _check_tensor


def swap_tensor(origin_tensor, transform=None, return_transform=False):
    r"""Swap the tensor with a new order of indices.

    This function will transform a tensor using a new order of indices,
    while the new order will not only swap between rows but also swap
    between columns.

    Parameters
    ----------
    origin_tensor : tensor, shape (n_samples, n_samples)
        The origin tensor to be swapped.
    transform : tensor, shape (n_samples), optional
        The transform used to swap the original tensor. If it is not
        provided, the transform will generate randomly.
    return_transform : bool, optional
        If return the transformation being performed.

    Returns
    -------
    transformed_tensor : tensor, shape (n_samples, n_samples)
        The transformed tensor swapped using new order of indices.
    """

    origin_tensor = _check_tensor(origin_tensor)

    if transform is None:
        transform = torch.randperm(origin_tensor.shape[0])

    transformed_tensor = \
        torch.take_along_dim(
            torch.take_along_dim(
                origin_tensor,
                transform[..., None],
                dim=0
            ),
            transform[None],
            dim=1
        )

    if return_transform:
        return (transformed_tensor, transform)
    else:
        return transformed_tensor


def replace_tensor(origin_tensor, origin_values, replace_values):
    r"""Replace the values inside a tensor with new values.

    This function will replace values inside the tensor with
    replacements.

    Parameters
    ----------
    origin_tensor : tensor
        The original tensor to be replaced.
    origin_values : list, tensor, shape (n_replacements, )
        The original values to be replaced inside the tensor.
    replace_values : list, tensor, shape (n_replacements, )
        The replacements values to replace the original values.

    Returns
    -------
    arr : tensor, shape (n_samples, )
        The replaced tensor.
    """

    origin_tensor = _check_tensor(origin_tensor)
    origin_values = _check_tensor(origin_values)
    replace_values = _check_tensor(replace_values)

    tensor = origin_tensor.clone().detach()

    tensor[
        tuple(
            map(
                torch.cat,
                zip(
                    *[
                        torch.where(tensor == origin_value)
                        for origin_value in origin_values
                    ]
                )
            )
        )
    ] = torch.cat(
        [
            torch.tile(
                replace_value.clone().detach(),
                torch.where(tensor == origin_value)[0].shape
            )
            for origin_value, replace_value
            in zip(origin_values, replace_values)
        ]
    )

    return tensor
