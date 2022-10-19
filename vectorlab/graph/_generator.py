"""
Traditional graph generators.
"""

import numpy as np

from scipy import sparse

from ._utils import to_bidirectional_graph
from ..utils._check import check_valid_int, check_valid_float


def graph_er_generator(n_nodes, density=0.5, ensure_bi=True):
    r"""Erdos-Renyi (ER) model is the simplest and most
    well-known generative model of graphs. In this model
    we define the likelihood of an edge occurring between
    any pair of nodes as

        .. math::
            P(\mathrm{A}[u, v] = 1) = r,
            \forall u, v \in \mathcal{V}, u \neq v

    where :math:`r \in [0, 1]` is a parameter controlling
    the density of the graph. In other words, the ER model
    simply assumes that the probability of an edge occurring
    between any pairs of nodes is equal to :math:`r`.

    Parameters
    ----------
    n_nodes : int
        The number of nodes in the generative graph.
    density : float
        The probability of an edge occurring.
    ensure_bi : bool
        If ensure to be a bidirectional graph.

    Returns
    -------
    adj_mat : scipy.sparse.coo_matrix
        The adjacency matrix of generative graph.
    """

    n_nodes = check_valid_int(
        n_nodes,
        lower=1,
        variable_name='n_nodes'
    )

    density = check_valid_float(
        density,
        lower=0.0, upper=1.0,
        variable_name='density'
    )

    row, col = np.triu_indices(n_nodes, k=1)
    data = np.random.binomial(1, p=density, size=row.shape[0])

    adj_mat = sparse.coo_matrix(
        (data[data != 0], (row[data != 0], col[data != 0])),
        shape=(n_nodes, n_nodes)
    )

    if ensure_bi:
        adj_mat = to_bidirectional_graph(adj_mat)

    return adj_mat


def graph_sbm_generator(n_nodes, n_clusters,
                        inner_density_range=(0.4, 1),
                        outer_density_range=(0, 0.5),
                        ensure_bi=True):
    r"""Many traditional graph generation approaches seek to improve the
    ER model by better capturing additional properties of real-world graphs,
    which the ER model ignores. One prominent example is the class of
    `stochastic block models (SBMs)`, which seek to generate graphs with
    community structure.

    In a basic `SBM` model, we specify a number :math:`\gamma` of different
    blocks: :math:`\mathcal{C}_1, \dots, \mathcal{C}_{\gamma}`. Every node
    :math:`u \in \mathcal{V}` then has a probability :math:`p_i` of belonging
    to block :math:`i`, i.e. :math:`p_i = P(u \in \mathcal{C}_i), \forall u \in
    \mathcal{V}, i = 1, \dots, \gamma` where :math:`\sum_{i=1}^{\gamma} p_i =
    1`. Edge probabilities are then specified by a block-to-block probability
    matrix :math:`\mathbf{C}  \in [0, 1]^{\gamma \times \gamma}`, where
    :math:`\mathcal{C}[i, j]` gives the probability of an edge occurring
    between a node in block :math:`\mathcal{C}_i` and a node in block
    :math:`\mathcal{C}_j`. The generative process for the basic `SBM` model is
    as follows:

    1. For every node :math:`u \in \mathcal{V}`, we assign :math:`u` to a block
    :math:`\mathcal{C}_i` by sampling from the categorical distribution defined
    by :math:`(p_i, \dots, p_{\gamma})`.

    2. For every pair of nodes :math:`u \in \mathcal{C}_i` and :math:`v \in
    \mathcal{C}_j` we sample an edge according to

        .. math::
            P(\mathbf{A}[u, v] = 1) = \mathbf{C}[i, j]


    Parameters
    ----------
    n_nodes : int
        The number of nodes in the generative graph.
    n_clusters : int
        The number of clusters in the generative graph.
    inner_density_range : tuple, optional
        The density of edges inside the same cluster. (lower, upper)
    outer_density_range : tuple, optional
        The density of edges between different clusters. (lower, upper)
    ensure_bi : bool, optional
        If ensure to be a bidirectional graph.

    Returns
    -------
    adj_mat : scipy.sparse.coo_matrix
        The adjacency matrix of generative graph.
    """

    n_nodes = check_valid_int(
        n_nodes,
        lower=1,
        variable_name='n_nodes'
    )

    n_clusters = check_valid_int(
        n_clusters,
        lower=1,
        variable_name='n_clusters'
    )

    row, col = np.triu_indices(n_clusters, k=1)
    cluster_density_matrix = sparse.coo_matrix(
        (
            np.random.uniform(
                outer_density_range[0],
                outer_density_range[1],
                row.shape[0]
            ),
            (row, col)
        ),
        shape=(n_clusters, n_clusters)
    )
    cluster_density_matrix += cluster_density_matrix.T

    row, col = np.diag_indices(n_clusters)
    cluster_density_diag_matrix = sparse.coo_matrix(
        (
            np.random.uniform(
                inner_density_range[0],
                inner_density_range[1],
                row.shape[0]
            ),
            (row, col)
        ),
        shape=(n_clusters, n_clusters)
    )
    cluster_density_matrix += cluster_density_diag_matrix
    # cluster_density_matrix = cluster_density_matrix.toarray()

    node_clusters = [np.random.choice(range(n_clusters))
                     for _ in range(n_nodes)]

    row, col = np.triu_indices(n_nodes, k=1)
    data = np.array([np.random.binomial(
        1,
        p=cluster_density_matrix[node_clusters[row_], node_clusters[col_]])
        for row_, col_ in zip(row, col)])

    adj_mat = sparse.coo_matrix(
        (data[data != 0], (row[data != 0], col[data != 0])),
        shape=(n_nodes, n_nodes)
    )

    if ensure_bi:
        adj_mat = to_bidirectional_graph(adj_mat)

    return adj_mat


def graph_pa_generator(n_nodes, n_init_nodes, ensure_bi=True):
    r"""Preferential attachment (PA) model attempts to capture this
    characteristic property of real-world degree distributions. The PA
    model is built around the assumptions that many real-world graphs
    exhibit `power law` degree distributions, meaning that the probability
    of a node :math:`u` having degree :math:`d_u` is roughly given by the
    following equation:

        .. math::
            P(d_u = k) \propto k^{- \alpha}

    where :math:`\alpha \gt 1` is a parameter. Power law distributions -
    and other related distributions - have the property that they are
    `heavy tailed`. Formally, being heavy tailed means that a probability
    distribution goes to zero for extreme values slower than an exponential
    distribution. This means that heavy-tailed distributions assign non-trivial
    probability mass to events that are essentially “impossible” under a
    standard exponential distribution. In the case of degree distributions,
    this heavy tailed nature essentially means that there is a non-zero chance
    of encountering a small number of very high-degree nodes. Intuitively,
    power law degree distributions capture the fact that real-world graphs
    have a large number of nodes with small degrees but also have a small
    number of nodes with extremely large degrees.

    The PA model generates graphs that exhibit power-law degree distributions
    using a simple generative process:

    1. First, we initialize  a fully connected graph with :math:`m_0` nodes

    2. Next, we iteratively add :math:`n - m_0` nodes to this graph. For each
    new node :math:`u` that we add at iteration :math:`t`, we connect it to
    :math:`m \lt m_0` existing nodes in the graph, and we choose its :math:`m`
    neighbors by sampling without replacement according to the following
    probability distribution:

        .. math::
            P(\mathbf{A}[u ,v]) = \frac{d_v^{(t)}}{\sum_{v' \in
            \mathcal{V}^{(t)}} d_{v'}^{(t)}}

    where :math:`d_v^{(t)}` denotes the degree of node :math:`v` at iteration
    :math:`t` and :math:`\mathcal{V}^{(t)}` denotes the set of nodes that have
    been added to the graph up to iteration :math:`t`.


    Parameters
    ----------
    n_nodes : int
        The number of nodes in the generative graph.
    n_init_nodes : int
        The number of initial nodes to set a fully connected graph.
    ensure_bi : bool, optional
        If ensure to be a bidirectional graph.

    Returns
    -------
    adj_mat : scipy.sparse.coo_matrix
        The adjacency matrix of generative graph.
    """

    n_nodes = check_valid_int(
        n_nodes,
        lower=1,
        variable_name='n_nodes'
    )

    n_init_nodes = check_valid_int(
        n_init_nodes,
        lower=1,
        variable_name='n_init_nodes'
    )

    adj_mat = np.zeros((n_nodes, n_nodes))

    for row in range(1, n_nodes):
        for col in range(row):
            if row < n_init_nodes and col < n_init_nodes:
                adj_mat[row][col] = 1
            else:
                adj_mat[row][col] = np.random.binomial(
                    1,
                    p=(np.sum(adj_mat[:, col]) / np.sum(adj_mat))
                )

        if np.sum(adj_mat[row, :]) == 0:
            adj_mat[row][np.random.choice(range(row))] = 1

    adj_mat = sparse.coo_matrix(adj_mat, shape=(n_nodes, n_nodes))

    if ensure_bi:
        adj_mat = to_bidirectional_graph(adj_mat)

    return adj_mat


def graph_mesh_generator(layers_nodes, ensure_bi=True):
    r"""The mesh generator of graph attempts to simulate the full mesh
    structure as real world network topology. We first define the number
    of nodes in each layer. And the nodes between two consecutive layers
    are fully connected.

    Parameters
    ----------
    layers_nodes : list
        The number of nodes in each layer, from bottom to top.
    ensure_bi : bool, optional
        If ensure to be a bidirectional graph.

    Returns
    -------
    adj_mat : scipy.sparse.coo_matrix
        The adjacency matrix of generative graph.
    """

    n_nodes = np.sum(layers_nodes)
    n_layers = len(layers_nodes)

    nodes = np.arange(n_nodes)
    layers = np.empty(0)

    for i, layer_node in enumerate(layers_nodes):
        layers = np.concatenate((layers, np.repeat(i, layer_node)))

    coo = np.empty((2, 0))
    for i in range(n_layers - 1):
        lower_nodes = nodes[layers == i]
        upper_nodes = nodes[layers == (i + 1)]

        coo = np.hstack(
            (
                coo,
                np.array(np.meshgrid(lower_nodes, upper_nodes)).reshape(2, -1)
            )
        )

    data = np.ones(coo.shape[1])

    adj_mat = sparse.coo_matrix(
        (data, coo),
        shape=(n_nodes, n_nodes)
    )

    if ensure_bi:
        adj_mat = to_bidirectional_graph(adj_mat)

    return adj_mat


def graph_random_generator(n_nodes, ensure_bi=True):
    r"""Randomly generate a graph.

    The random generator just simply generates a graph in a fully
    random manner. For each pair of nodes, random generator just
    randomly decide whether there is a line between them or not.

    Parameters
    ----------
    n_nodes : int
        The number of nodes in the generative graph.
    ensure_bi : bool
        If ensure to be a bidirectional graph.

    Returns
    -------
    adj_mat : scipy.sparse.coo_matrix
        The adjacency matrix of generative graph.
    """

    n_nodes = check_valid_int(
        n_nodes,
        lower=1,
        variable_name='n_nodes'
    )

    row, col = np.triu_indices(n_nodes, k=1)
    adj_mat = sparse.coo_matrix(
        (
            np.random.randint(0, 2, row.shape[0]),
            (row, col)
        ),
        shape=(n_nodes, n_nodes)
    )

    if ensure_bi:
        adj_mat = to_bidirectional_graph(adj_mat)

    adj_mat += sparse.eye(n_nodes, dtype=np.int_)
    adj_mat = adj_mat.tocoo()

    return adj_mat
