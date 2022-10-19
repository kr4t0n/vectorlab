import pytest
import numpy as np

from scipy import sparse

from vectorlab.graph.walking import bfs, dfs, random_walk


@pytest.mark.parametrize('p', [0.1, 0.3, 0.5, 0.7, 0.9])
def test_bfs(p):
    n_nodes = 1000

    mat = np.random.binomial(1, p=p, size=(n_nodes, n_nodes))
    coo_mat = sparse.coo_matrix(mat)

    start_node = 0
    end_node = n_nodes // 2

    path = bfs(coo_mat, start_node, end_node)

    assert (path is None or (path[0] == start_node and path[-1] == end_node))


def bfs_efficency():
    n_nodes = 1000

    mat = np.random.binomial(1, p=0.3, size=(n_nodes, n_nodes))
    coo_mat = sparse.coo_matrix(mat)

    start_node = 0
    end_node = n_nodes // 2

    bfs(coo_mat, start_node, end_node)


def test_bfs_efficency(benchmark):
    benchmark(bfs_efficency)


@pytest.mark.parametrize('p', [0.1, 0.3, 0.5, 0.7, 0.9])
def test_dfs(p):
    n_nodes = 1000

    mat = np.random.binomial(1, p=p, size=(n_nodes, n_nodes))
    coo_mat = sparse.coo_matrix(mat)

    start_node = 0
    end_node = n_nodes // 2

    path = dfs(coo_mat, start_node, end_node)

    assert (path is None or (path[0] == start_node and path[-1] == end_node))


def dfs_efficency():
    n_nodes = 1000

    mat = np.random.binomial(1, p=0.3, size=(n_nodes, n_nodes))
    coo_mat = sparse.coo_matrix(mat)

    start_node = 0
    end_node = n_nodes // 2

    dfs(coo_mat, start_node, end_node)


def test_dfs_efficency(benchmark):
    benchmark(dfs_efficency)


@pytest.mark.parametrize('p', [0.5, 0.7, 0.9])
@pytest.mark.parametrize('start_node', [0, 500, 999])
@pytest.mark.parametrize('path_length', [10, 100, 500])
def test_random_walk(p, start_node, path_length):
    n_nodes = 1000

    mat = np.random.binomial(1, p=p, size=(n_nodes, n_nodes))
    coo_mat = sparse.coo_matrix(mat)

    path = random_walk(coo_mat, start_node, path_length)

    assert path[0] == start_node
    assert len(path) == path_length


def random_walk_efficency():
    n_nodes = 1000

    mat = np.random.binomial(1, p=0.3, size=(n_nodes, n_nodes))
    coo_mat = sparse.coo_matrix(mat)

    start_node = 0
    path_length = 100

    random_walk(coo_mat, start_node, path_length)


def test_random_walk_efficency(benchmark):
    benchmark(random_walk_efficency)
