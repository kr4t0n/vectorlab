import torch

from vectorlab.vector import swap_tensor, replace_tensor


def test_swap_tensor():

    n_samples = 10
    mat = torch.rand(n_samples, n_samples)

    # Transformation
    trans_mat, trans = swap_tensor(mat, return_transform=True)

    # Reversing the transformation
    reversed_trans = replace_tensor(
        torch.arange(n_samples),
        trans,
        torch.arange(n_samples)
    )
    reversed_mat = swap_tensor(trans_mat, transform=reversed_trans)

    assert torch.all(mat == reversed_mat)


def swap_tensor_efficency():

    n_samples = 10
    mat = torch.rand(n_samples, n_samples)

    swap_tensor(mat, return_transform=True)


def test_swap_tensor_efficency(benchmark):

    benchmark(swap_tensor_efficency)


def test_replace_tensor():

    n_samples = 10
    arr = torch.randint(0, 2, (n_samples, ))

    trans_arr = replace_tensor(arr, [0, 1], [1, 0])
    reversed_arr = replace_tensor(trans_arr, [0, 1], [1, 0])

    assert torch.all(arr == reversed_arr)


def replace_tensor_efficency():

    n_samples = 10
    arr = torch.randint(0, 2, (n_samples, ))

    replace_tensor(arr, [0, 1], [1, 0])


def test_replace_tensor_efficency(benchmark):

    benchmark(replace_tensor_efficency)
