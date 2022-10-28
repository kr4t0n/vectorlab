import pytest
import numpy as np

from vectorlab.vector import (
    swap_matrix,
    find_forward_transform,
    find_backward_transform,
    replace_array
)


def test_swap_matrix():

    n_samples = 10
    mat = np.random.rand(n_samples, n_samples)

    # Transformation
    trans_mat, trans = swap_matrix(mat, return_transform=True)

    # Reversing the transformation
    reversed_trans = replace_array(
        np.arange(n_samples),
        trans,
        np.arange(n_samples)
    )
    reversed_mat = swap_matrix(trans_mat, transform=reversed_trans)

    assert np.all(mat == reversed_mat)


def swap_matrix_efficency():

    n_samples = 10
    mat = np.random.rand(n_samples, n_samples)

    swap_matrix(mat, return_transform=True)


def test_swap_matrix_efficency(benchmark):

    benchmark(swap_matrix_efficency)


@pytest.mark.parametrize('k', [10, 30, 50])
@pytest.mark.parametrize('method', ['unary', 'pairwise', 'lawler', 'yaaq'])
@pytest.mark.parametrize('rerun', [1000])
def test_find_foward_transform(k, method, rerun):

    success_number = 0

    for _ in range(rerun):

        mat = np.zeros((k, k))

        mat[np.tril_indices(k, -1)] = \
            np.random.randint(0, 2, int(k * (k - 1) / 2))
        mat = mat + mat.T
        mat[np.diag_indices(k)] = 1

        trans_mat = swap_matrix(mat)

        forward_transform = find_forward_transform(
            mat, trans_mat, method=method
        )

        if np.all(swap_matrix(mat, forward_transform) == trans_mat):
            success_number += 1

    print(
        f'Method: {method}, node size: {k}, '
        f'success_rate: {success_number * 1.0 / rerun:.2%}'
    )


def find_forward_transform_efficency(method):

    k = 30

    mat = np.zeros((k, k))

    mat[np.tril_indices(k, -1)] = \
        np.random.randint(0, 2, int(k * (k - 1) / 2))
    mat = mat + mat.T
    mat[np.diag_indices(k)] = 1

    trans_mat = swap_matrix(mat)

    find_forward_transform(mat, trans_mat, method=method)


@pytest.mark.parametrize('method', ['unary', 'pairwise', 'lawler', 'yaaq'])
def test_find_forward_transform_efficency(method, benchmark):
    benchmark.pedantic(
        find_forward_transform_efficency,
        kwargs={'method': method},
        rounds=100
    )


@pytest.mark.parametrize('k', [10, 30, 50])
@pytest.mark.parametrize('method', ['unary', 'pairwise', 'lawler', 'yaaq'])
@pytest.mark.parametrize('rerun', [1000])
def test_find_backward_transform(k, method, rerun):

    success_number = 0

    for _ in range(rerun):

        mat = np.zeros((k, k))

        mat[np.tril_indices(k, -1)] = \
            np.random.randint(0, 2, int(k * (k - 1) / 2))
        mat = mat + mat.T
        mat[np.diag_indices(k)] = 1

        trans_mat = swap_matrix(mat)

        forward_transform = find_backward_transform(
            mat, trans_mat, method=method
        )

        if np.all(swap_matrix(trans_mat, forward_transform) == mat):
            success_number += 1

    print(
        f'Method: {method}, node size: {k}, '
        f'success_rate: {success_number * 1.0 / rerun:.2%}'
    )


def find_backward_transform_efficency(method):

    k = 30

    mat = np.zeros((k, k))

    mat[np.tril_indices(k, -1)] = \
        np.random.randint(0, 2, int(k * (k - 1) / 2))
    mat = mat + mat.T
    mat[np.diag_indices(k)] = 1

    trans_mat = swap_matrix(mat)

    find_backward_transform(mat, trans_mat, method=method)


@pytest.mark.parametrize('method', ['unary', 'pairwise', 'lawler', 'yaaq'])
def test_find_backward_transform_efficency(method, benchmark):
    benchmark.pedantic(
        find_backward_transform_efficency,
        kwargs={'method': method},
        rounds=100
    )


def test_replace_array():

    n_samples = 10
    arr = np.random.randint(0, 2, n_samples)

    trans_arr = replace_array(arr, [0, 1], [1, 0])
    reversed_arr = replace_array(trans_arr, [0, 1], [1, 0])

    assert np.all(arr == reversed_arr)


def replace_array_efficency():

    n_samples = 10
    arr = np.random.randint(0, 2, n_samples)

    replace_array(arr, [0, 1], [1, 0])


def test_replace_array_efficency(benchmark):

    benchmark(replace_array_efficency)
