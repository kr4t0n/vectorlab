"""
BFS is a graph walking algorithm standing for
breadth first searching.
"""

from ...base import Queue
from ...utils._check import check_valid_int


def bfs(adj_mat, start_node, end_node, return_visited=False):
    r"""Breadth-first search (BFS) is an algorithm for traversing
    or searching tree or graph data structure. It starts from an
    arbitrary tree node and explores all of the neighbor nodes at
    the present depth prior to moving on the nodes at the next
    depth level.

    Parameters
    ----------
    adj_mat : array_like, scipy.sparse.spmatrix, shape (n_nodes, n_nodes)
        The adjacency matrix of a graph.
    start_node : int
        The node index to start searching.
    end_node : int
        The node index to stop searching.
    return_visited : bool, optional
        If return the visited nodes.

    Returns
    -------
    path : list
        The path from start node to end node using BFS algorithm. If
        there is no path from start node to end node, a None value is
        returned.
    visited : list
        The order of nodes being visited.
    """

    start_node = check_valid_int(
        start_node,
        lower=0, variable_name='start_node'
    )
    end_node = check_valid_int(
        end_node,
        lower=0, variable_name='end_node'
    )

    path_queue = Queue()
    path_queue.push([start_node])

    visited = [start_node]

    while path_queue:
        path = path_queue.pop()

        current_node = path[-1]

        if current_node == end_node:
            return (path, visited) if return_visited else path
        else:
            next_hops = adj_mat.getrow(current_node).indices

            next_hops_unvisited = set(next_hops) - set(visited)

            for next_hop in next_hops_unvisited:
                path_queue.push(path + [next_hop])
                visited.append(next_hop)

    return (None, visited) if return_visited else None
