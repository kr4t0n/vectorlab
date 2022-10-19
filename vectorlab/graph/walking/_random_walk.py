"""
Random Walk is a graph walking algorithm. It is
often used to observe the structure of a graph.
"""

import numpy as np

from ...utils._check import check_valid_int


def random_walk(adj_mat, start_node, path_length):
    r"""Random walk is an algorithm for traversing and searching
    tree or graph structure. It starts from an arbitrary tree node
    and explores the next node randomly, which is controlled by
    the length of return path.

    Parameters
    ----------
    adj_mat : array_like, scipy.sparse.spmatrix, shape (n_nodes, n_nodes)
        The adjacency matrix of a graph.
    start_node : int
        The node index to start searching.
    path_length : int
        The length of return path.

    Returns
    -------
    path : list
        The path from start node using random walk.
    """

    start_node = check_valid_int(
        start_node,
        lower=0, variable_name='start_node'
    )
    path_length = check_valid_int(
        path_length,
        lower=0, variable_name='path_length'
    )

    path = [start_node]

    while len(path) < path_length:
        current_node = path[-1]

        next_hops = adj_mat.getrow(current_node).indices
        next_hop = np.random.choice(next_hops)

        path.append(next_hop)

    return path
