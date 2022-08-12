import pytest
import numpy as np

from vectorlab.graph import graph_er_generator
from vectorlab.graph import (
    add_self_loop,
    remove_self_loop,
    to_bidirectional_graph,
    is_connectivity,
    make_connectivity,
    find_subgraphs,
    find_min_degree_nodes,
    find_max_degree_nodes,
    make_k_core,
    k_core_decomposition,
    get_node_ordering
)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_add_self_loop(n_nodes):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    adj_mat = remove_self_loop(adj_mat)
    adj_mat = add_self_loop(adj_mat)

    assert np.all(adj_mat.diagonal() == 1)


def add_self_loop_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    adj_mat = add_self_loop(adj_mat)


def test_add_self_loop_efficiency(benchmark):
    benchmark(add_self_loop_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_remove_self_loop(n_nodes):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    adj_mat = remove_self_loop(adj_mat)

    assert np.all(adj_mat.diagonal() == 0)


def remove_self_loop_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    remove_self_loop(adj_mat)


def test_remove_self_loop_efficiency(benchmark):
    benchmark(remove_self_loop_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_to_bidirectional_graph(n_nodes):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=False)

    assert (adj_mat + adj_mat.T != adj_mat * 2).nnz != 0

    adj_mat = to_bidirectional_graph(adj_mat)

    assert (adj_mat + adj_mat.T != adj_mat * 2).nnz == 0


def to_bidirectional_graph_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=False)
    adj_mat = to_bidirectional_graph(adj_mat)


def test_to_bidirectional_graph_efficiency(benchmark):
    benchmark(to_bidirectional_graph_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
@pytest.mark.parametrize('density', [0, 1])
def test_is_connectivity(n_nodes, density):

    adj_mat = graph_er_generator(n_nodes, density=density, ensure_bi=True)

    if density == 0:
        assert not is_connectivity(adj_mat)
    elif density == 1:
        assert is_connectivity(adj_mat)


def is_connectivity_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    is_connectivity(adj_mat)


def test_is_connectivity_efficiency(benchmark):
    benchmark(is_connectivity_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_make_connectivity(n_nodes):

    adj_mat = graph_er_generator(n_nodes, density=0, ensure_bi=True)
    adj_mat = make_connectivity(adj_mat)

    assert is_connectivity(adj_mat)


def make_connectivity_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    make_connectivity(adj_mat)


def test_make_connectivity_efficiency(benchmark):
    benchmark(make_connectivity_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
@pytest.mark.parametrize('method', ['bfs', 'dfs'])
def test_find_subgraphs(n_nodes, method):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    subgraphs = find_subgraphs(adj_mat, n_nodes, method=method)

    for subgraph in subgraphs:

        mask = np.zeros(n_nodes, dtype=np.bool_)
        mask[subgraph] = True

        assert is_connectivity(adj_mat.tolil()[mask, mask].tocsr())


def find_subgraphs_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    find_subgraphs(adj_mat)


def test_find_subgraphs_efficiency(benchmark):
    benchmark(find_subgraphs_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_find_min_degree_nodes(n_nodes):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    min_degree_nodes = find_min_degree_nodes(adj_mat)

    degs = adj_mat.sum(axis=0).A1

    assert np.all(degs[min_degree_nodes] == degs.min())


def find_min_degree_nodes_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    find_min_degree_nodes(adj_mat)


def test_find_min_degree_nodes_efficiency(benchmark):
    benchmark(find_min_degree_nodes_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_find_max_degree_nodes(n_nodes):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    max_degree_nodes = find_max_degree_nodes(adj_mat)

    degs = adj_mat.sum(axis=0).A1

    assert np.all(degs[max_degree_nodes] == degs.max())


def find_max_degree_nodes_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    find_max_degree_nodes(adj_mat)


def test_find_max_degree_nodes_efficiency(benchmark):
    benchmark(find_max_degree_nodes_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
@pytest.mark.parametrize('k', [2, 4])
def test_make_k_core(n_nodes, k):

    adj_mat = graph_er_generator(n_nodes, density=1, ensure_bi=True)
    removal_indices, remaining_indices, k_adj_mat = \
        make_k_core(adj_mat, k, n_nodes, return_k_adj_mat=True)

    assert removal_indices.shape[0] + remaining_indices.shape[0] == n_nodes

    degs = k_adj_mat.sum(axis=0).A1

    assert np.all(degs[degs != 0] >= k)


def make_k_core_efficiency(n_nodes=1000, k=5):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    make_k_core(adj_mat, k, n_nodes, return_k_adj_mat=True)


def test_make_k_core_efficiency(benchmark):
    benchmark(make_k_core_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
def test_k_core_decomposition(n_nodes):

    adj_mat = graph_er_generator(n_nodes, density=1, ensure_bi=True)
    decomposition = k_core_decomposition(adj_mat, n_nodes)

    removal_indices = []
    for k, v in decomposition.items():
        removal_indices.extend(v.tolist())

        k_removal_indices, _ = make_k_core(
            adj_mat, k + 1, n_nodes, return_k_adj_mat=False
        )

        assert np.all(np.sort(removal_indices) == np.sort(k_removal_indices))


def k_core_decomposition_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    k_core_decomposition(adj_mat, n_nodes)


def test_k_core_decomposition_efficiency(benchmark):
    benchmark(k_core_decomposition_efficiency)


@pytest.mark.parametrize('n_nodes', [10, 100])
@pytest.mark.parametrize('method', ['bfs', 'dfs', 'k-core'])
def test_get_node_ordering(n_nodes, method):

    adj_mat = graph_er_generator(n_nodes, density=1, ensure_bi=True)
    node_ordering = get_node_ordering(adj_mat, n_nodes, method=method)

    assert np.all(np.sort(node_ordering) == np.arange(n_nodes))


def get_node_ordering_efficiency(n_nodes=1000):

    adj_mat = graph_er_generator(n_nodes, ensure_bi=True)
    get_node_ordering(adj_mat, n_nodes, method='k-core')


def test_get_node_ordering_efficiency(benchmark):
    benchmark(get_node_ordering_efficiency)
