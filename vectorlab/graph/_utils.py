"""
Several useful utilities over graph data are provided.
"""

import random
import numpy as np

from scipy import sparse
from copy import deepcopy
from itertools import combinations, groupby

from .walking._bfs import bfs
from .walking._dfs import dfs
from ..utils._check import check_valid_int


def add_self_loop(adj_mat):
    r"""Add self loop to adjacency matrix.

    We simply set the diagonal elements into zeros.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of the graph.

    Returns
    -------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of the graph.
    """

    return_adj_mat = deepcopy(adj_mat).asformat('lil')

    return_adj_mat.setdiag(1)

    return_adj_mat = return_adj_mat.asformat(
        adj_mat.getformat()
    )

    return return_adj_mat


def remove_self_loop(adj_mat):
    r"""Remove the existed self loop from adjacency matrix.

    First, setting all diagonal elements into zeros, then eliminating
    all zeros happened in the adjacency matrix to preserve the
    concision of it.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of the graph.

    Returns
    -------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of the graph.
    """

    return_adj_mat = deepcopy(adj_mat).asformat('lil')

    return_adj_mat.setdiag(0)

    return_adj_mat = return_adj_mat.asformat(
        adj_mat.getformat()
    )
    return_adj_mat.eliminate_zeros()

    return return_adj_mat


def to_bidirectional_graph(adj_mat):
    r"""Ensure the generated graph is a bidirectional graph

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative graph.

    Returns
    -------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative bidirectional graph.
    """

    return_adj_mat = deepcopy(adj_mat)

    return_adj_mat = return_adj_mat + return_adj_mat.T
    return_adj_mat.data[:] = 1

    return return_adj_mat


def to_deterministic_graph(adj_mat, thres=0.5,
                           remain_prob=False,
                           ensure_connective=True,
                           ensure_bidirectional=True):
    r"""Change a probabilistic graph to a deterministic graph.

    The adjacency matrix of some generated graph procedure can represent
    the edge probabilities. This function will change these edge probabilities
    to a determined zeros or ones using provided threshold.

    In the transformation, `thres` served as a parameter to control the
    result density, while `ensure_connective` will ensure that every node is
    connected to at least one other nodes, and `ensure_bidirectional` will
    ensure the return adjacency matrix is bidirectional graph.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative graph.
    thres : float, optional
        The threshold to determine whether an edge should appear or not.
    remain_prob : bool, optional
        If the remaining edges still have their probabilities.
    ensure_connective : bool, optional
        If every node has to be connected to at least one other node.
    ensure_bidirectional : bool, optional
        It return adjacency matrix is a bidirectional graph.

    Returns
    -------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative graph.
    """

    adj_mat = remove_self_loop(adj_mat)

    if ensure_connective:
        adj_mat = adj_mat.tocsr()

        adj_mat[np.arange(adj_mat.shape[0]),
                adj_mat.argmax(axis=1).ravel()] = \
            np.ones(adj_mat.shape[0])

        adj_mat = adj_mat.tocoo()

    adj_mat.data[adj_mat.data < thres] = 0
    adj_mat.eliminate_zeros()

    if not remain_prob:
        adj_mat.data[adj_mat.data >= thres] = 1

    adj_mat = add_self_loop(adj_mat)

    if ensure_bidirectional:
        adj_mat = to_bidirectional_graph(adj_mat)

    return adj_mat


def is_connectivity(adj_mat, n_nodes=None, method='dfs'):
    r"""Test a graph is connectivity graph or not.

    For a undirected graph, to judge whether a graph is a connectivity
    graph or not. We start from a random point, and to check whether
    if can visit any other points in the graph using graph searching
    method. To take advantage of BFS, DFS in graph searching, we hack
    a nonexistent point, to find a path to that point. Obviously, there
    will not be a path to that point, however, the find process will
    explore every connected points possible. Therefore, we can only
    check whether the number of explored points is equal to the number
    of nodes in the graph.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative graph.
    n_nodes : int, optional
        The number of nodes in the graph. If not specified, it will use
        the shape[0] of adj_mat.
    method : str, optional
        The method to be used in graph searching algorithm. Currently,
        methods supported are BFS and DFS.

    Returns
    -------
    bool
        If the graph is connectivity graph or not.

    Raises
    ------
    ValueError
        If the method specified to test connectivity is not one of the BFS
        or DFS, a ValueError is raised.
    """

    if n_nodes is None:
        n_nodes = adj_mat.shape[0]
    else:
        n_nodes = check_valid_int(n_nodes, lower=0, variable_name='n_nodes')

    start_node = 0
    # Hack a nonexistent node
    end_node = n_nodes

    if method == 'bfs':
        _, visited = bfs(adj_mat, start_node, end_node,
                         return_visited=True)
    elif method == 'dfs':
        _, visited = dfs(adj_mat, start_node, end_node,
                         return_visited=True)
    else:
        raise ValueError(
            f'Currently does not support to use method {method} '
            f'to decide connectivity.'
        )

    if len(visited) == n_nodes:
        return True
    else:
        return False


def make_connectivity(adj_mat, n_nodes=None):
    r"""Make a graph become connected.

    Random add edges between pairs of sub-graphs, making the big
    graph become connected.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative graph.
    n_nodes : int, optional
        The number of nodes in the graph. If not specified, it will use
        the shape[0] of adj_mat.

    Returns
    -------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of connected graph.
    """

    if n_nodes is None:
        n_nodes = adj_mat.shape[0]
    else:
        n_nodes = check_valid_int(n_nodes, lower=0, variable_name='n_nodes')

    subgraphs = dict(enumerate(find_subgraphs(adj_mat, n_nodes)))
    subgraphs_combs = combinations(subgraphs.keys(), r=2)

    for _, combs in groupby(subgraphs_combs, key=lambda x: x[0]):

        combs = list(combs)
        random_comp = random.choice(combs)

        source = random.choice(list(subgraphs[random_comp[0]]))
        target = random.choice(list(subgraphs[random_comp[1]]))

        adj_mat += sparse.coo_matrix(
            (
                [1, 1],
                (
                    [source, target],
                    [target, source]
                )
            ),
            shape=(n_nodes, n_nodes)
        )

    return adj_mat


def find_subgraphs(adj_mat, n_nodes=None, method='dfs'):
    r"""Find all the subgraphs in the given graph.

    Finding subgraphs is quite similar to check connectivity. We start
    from a random point in the remaining points, and to explore every
    possible points it can visit by exploiting a nonexistent point, and
    to find a path to that point. Once we retrieve all the possible could
    be visited points, we subtract these points from the remaining points,
    and start the iteration again until the remaining points are empty,
    indicating that we have visited all points in the single graph. Each
    iteration result contributes to a subgraph.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generative graph.
    n_nodes : int, optional
        The number of nodes in the graph. If not specified, it will use
        the shape[0] of adj_mat.
    method : str, optional
        The method to be used in graph searching algorithm. Currently,
        methods supported are BFS and DFS.

    Returns
    -------
    subgraphs : list
        The list of subgraphs, each element is a set of nodes indices.

    Raises
    ------
    ValueError
        If the method specified to test connectivity is not one of the BFS
        or DFS, a ValueError is raised.
    """

    if n_nodes is None:
        n_nodes = adj_mat.shape[0]
    else:
        n_nodes = check_valid_int(n_nodes, lower=0, variable_name='n_nodes')

    subgraphs = []
    remaining_nodes = set(np.arange(n_nodes))

    while remaining_nodes:
        start_node = np.random.choice(list(remaining_nodes), 1)[0]
        # Hack a nonexistent node
        end_node = n_nodes

        if method == 'bfs':
            _, visited = bfs(adj_mat, start_node, end_node,
                             return_visited=True)
        elif method == 'dfs':
            _, visited = dfs(adj_mat, start_node, end_node,
                             return_visited=True)
        else:
            raise ValueError(
                f'Currently does not support to use method {method} '
                f'to decide connectivity.'
            )

        subgraphs.append(visited)
        remaining_nodes -= set(visited)

    return subgraphs


def find_min_degree_nodes(adj_mat):
    r"""Find all the nodes with minimum degree.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generated graph.

    Returns
    -------
    degs_min_indices : np.ndarray
        The nodes indices with minimum degree.
    """

    degs = adj_mat.sum(axis=0).A1
    degs_min_indices = np.flatnonzero(degs == degs.min())

    return degs_min_indices


def find_max_degree_nodes(adj_mat):
    r"""Find all the nodes with maximum degree.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of generated graph.

    Returns
    -------
    degs_max_indices : np.ndarray
        The node indices with maximum degree.
    """

    degs = adj_mat.sum(axis=0).A1
    degs_max_indices = np.flatnonzero(degs == degs.max())

    return degs_max_indices


def make_k_core(adj_mat, k, n_nodes=None, return_k_adj_mat=False):
    r"""Making a graph become a k-core sub-graph.

    Making a graph become a k-core sub-graph whose node degrees are all
    at least k.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of input graph.
    k : int
        The value of k-core to be found.
    n_nodes : None, optional
        The number of nodes in the graph. If not specified, it will use
        the shape[0] of adj_mat.
    return_k_adj_mat : bool, optional
        If return the k-core sub-graph adjacency matrix.

    Returns
    -------
    removal_indices : np.ndarray
        The node indices removed to make a k-core sub-graph.
    remaining_indices : np.ndarray
        The node indices remained to compose a k-core sub-graph.
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of k-core sub-graph.
    """

    k = check_valid_int(k, lower=1, variable_name='k-core')

    if n_nodes is None:
        n_nodes = adj_mat.shape[0]
    else:
        n_nodes = check_valid_int(n_nodes, lower=0, variable_name='n_nodes')

    removal_indices = []

    while True:

        degs = adj_mat.sum(axis=0).A1

        if degs.max() > 0 and degs[degs != 0].min() < k:

            indices = np.flatnonzero((degs > 0) & (degs < k))
            removal_indices.append(indices.tolist())

            mask = np.ones(adj_mat.shape[0], dtype=np.int_)
            mask[indices] = 0
            T = sparse.diags(mask, dtype=np.int_)

            adj_mat = T * adj_mat * T
        else:
            break

    if len(removal_indices) == 0:
        removal_indices = np.array([], dtype=np.int_)
    else:
        removal_indices = np.concatenate(removal_indices)
    remaining_indices = np.flatnonzero(degs > 0)

    if return_k_adj_mat:
        return removal_indices, remaining_indices, adj_mat
    else:
        return removal_indices, remaining_indices


def k_core_decomposition(adj_mat, n_nodes=None):
    r"""Decompose the graph in a k-core fashion.

    Based on the largest core number per node, we can uniquely
    determine a partition of all nodes, i.e., disjoint sets of
    nodes which share the same largest core number. We then
    assign the core number of each disjoint set by the largest
    core number of its nodes.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of input graph.
    n_nodes : None, optional
        The number of nodes in the graph. If not specified, it will use
        the shape[0] of adj_mat.

    Returns
    -------
    decomposition : dict
        The decomposition has keys of core number and values of corresponding
        nodes.
    """

    if n_nodes is None:
        n_nodes = adj_mat.shape[0]
    else:
        n_nodes = check_valid_int(n_nodes, lower=0, variable_name='n_nodes')

    k = 0
    decomposition = {}

    while True:
        k += 1

        removal_indices, remaining_indices, adj_mat = make_k_core(
            adj_mat, k,
            n_nodes=n_nodes,
            return_k_adj_mat=True
        )

        if removal_indices.size:
            decomposition[k - 1] = removal_indices

        if not remaining_indices.size:
            break

    return decomposition


def get_node_ordering(adj_mat, n_nodes=None, method='bfs'):
    r"""Get node ordering for a given graph using specified method.

    Since the node inside a given graph has no order, since the representation
    of a graph will not be changed with different orders. However, in order to
    standardize the order of nodes, thus to compare the similarity of two
    graphs easily, we could order the nodes using some specified methods.
    The methods include some classic searching algorithm, i.e., bfs and dfs
    from the highest degree of node, or using k-core decomposition to
    decompose graph into disjoint parts, and order the nodes by their degree
    inside each partition.

    Parameters
    ----------
    adj_mat : scipy.sparse.spmatrix
        The adjacency matrix of input graph.
    n_nodes : None, optional
        The number of nodes in the graph. If not specified, it will use
        the shape[0] of adj_mat.
    method : str, optional
        The method specified to order the nodes inside, currently supports
        bfs, dfs and k-core.

    Returns
    -------
    node_ordering : np.ndarray
        The node ordering generated by specified method.
    """

    if n_nodes is None:
        n_nodes = adj_mat.shape[0]
    else:
        n_nodes = check_valid_int(n_nodes, lower=0, variable_name='n_nodes')

    max_degree_nodes = find_max_degree_nodes(adj_mat)
    start_node = np.random.choice(max_degree_nodes)

    if method == 'bfs':
        _, node_ordering = bfs(
            adj_mat, start_node, n_nodes,
            return_visited=True
        )
    elif method == 'dfs':
        _, node_ordering = dfs(
            adj_mat, start_node, n_nodes,
            return_visited=True
        )
    elif method == 'k-core':
        degs = adj_mat.sum(axis=0).A1
        decomposition = k_core_decomposition(adj_mat, n_nodes)

        node_ordering = [
            decomposed_nodes[np.argsort(degs[decomposed_nodes])].tolist()
            for decomposed_nodes in decomposition.values()
        ]

        node_ordering = np.flip(np.concatenate(node_ordering))

    return node_ordering
