"""
The functions are focused on plotting relationship between variables.
"""

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse

from ._palettes import load_palette
from ..utils._check import _check_ndarray


def init_plot(width=10, height=8, dpi=100,
              ax_labels=None,
              title=None,
              despine=True, style='whitegrid'):
    r"""Initalize a plot, this will create a new figure.

    Parameters
    ----------
    width : int, optional
        The width of initialized figure.
    height : int, optional
        The height of initialized figure.
    dpi : int, optional
        The dpi of initialized figure.
    ax_labels : list, optional
        The coordinate labels used in initialized axes.
        When specified, it has to be a list contained of two names,
        for `x` and `y` coordinate respectfully.
    title : str, optional
        The title of plot figure.
    despine : bool, optional
        Whether initialized figure is despined or not.
    style : str, optional
        The style used to initialize figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The initialized figure.
    """

    # clear and close any existing figures
    plt.clf()
    plt.close()

    if style is not None:
        sns.set_style(style)

    fig = plt.figure(figsize=(width, height), dpi=dpi)

    n, e, w, s = (0.85, 0.9, 0.1, 0.15)
    fig.subplots_adjust(bottom=s, right=e, left=w, top=n)

    if ax_labels is not None:

        if len(ax_labels) != 2:
            raise ValueError(
                f'Invalid ax_labels, it needs two elements inside '
                f'for x and y axis. Current ax_labels have {len(ax_labels)} '
                f'elements.'
            )

        fig.supxlabel(ax_labels[0], fontdict={'size': 10})
        fig.supylabel(ax_labels[1], fontdict={'size': 10})

    if title:
        fig.suptitle(title)

    if despine:
        sns.despine(left=True, bottom=True)

    return fig


def plot2d(x, y, categories,
           cats=None,
           ax_pos=None, ax_labels=None,
           title=None, caption=None,
           markers='o', marker_sizes=1, lines='None',
           legendary=True, legend_title='None',
           palette='default'):
    r"""This will plot a two dimensional graph as desired.
    It will first retrieve each category inside categories,
    if cats is specified, the cats will be used instead.
    This plot function will plot each point in order of different
    categories.

    Parameters
    ----------
    x : array_like, shape (n_samples)
        `x` coordinate values for all points.
    y : array_like, shape (n_samples)
        `y` coordinate values for all points.
    categories : array_like, shape (n_samples)
        Categories for all points.
    cats : list, optional
        The cats used to plot, this can make plotting in order
        of different categories in order defined in cats.
    ax_pos : tuple, optional
        The positions to plot graph, provided if subplot is desired.
    ax_labels : list, optional
        The coordinate labels used in initialized axes.
        When specified, it has to be a list contained of two names,
        for `x` and `y` coordinate respectfully.
    title : str, optional
        The title of plot axes.
    caption : str, optional
        The caption of plot axes.
    markers : str, optional
        The marker used to plot the points.
    marker_sizes : int, optional
        The size of maker used to plot the points.
    lines : str, optional
        The line style used to connect points in the same category.
        If lines is 'None', no line will be plotted.
    legendary : bool, optional
        Whether plot legend information.
    legend_title : str, optional
        The legend title used to specify legend,
        when legendary is True.
    palette : str, optional
        The palette name used to assign difference color to
        different category.

    Returns
    -------
    ax : matplotlib.axes
        The axes of plotting.
    """

    x = _check_ndarray(x)
    y = _check_ndarray(y)
    categories = _check_ndarray(categories)

    if ax_pos is not None:
        ax = plt.subplot(*ax_pos)
    else:
        ax = plt.subplot(*(1, 1, 1))

    if ax_labels is not None:

        if len(ax_labels) != 2:
            raise ValueError(
                f'Invalid ax_labels, it needs two elements inside '
                f'for x and y axis. Current ax_labels have {len(ax_labels)} '
                f'elements.'
            )

        ax.set_xlabel(ax_labels[0], fontdict={'size': 10})
        ax.set_ylabel(ax_labels[1], fontdict={'size': 10})

    if title:
        ax.set_title(label=title, fontdict={'size': 12})

    if caption:
        ax.text(
            s='\n' + caption,
            horizontalalignment='left',
            x=0, y=-0.075,
            transform=ax.transAxes,
            fontdict={'size': 8}
        )

    if cats is None:
        cats, indices = np.unique(categories, return_index=True)
        cats = cats[np.argsort(indices)]

    if not isinstance(markers, dict):
        markers = {cat: markers for cat in cats}
    if not isinstance(marker_sizes, dict):
        marker_sizes = {cat: marker_sizes for cat in cats}
    if not isinstance(lines, dict):
        lines = {cat: lines for cat in cats}

    palette = load_palette(palette)
    if np.issubdtype(cats.dtype, np.integer):
        colors = {
            cat: palette[cat % len(palette)]
            for cat in cats
        }
    else:
        colors = {
            cat: palette[idx % len(palette)]
            for idx, cat in enumerate(cats)
        }

    for cat in cats:
        ax.plot(
            x[categories == cat],
            y[categories == cat],
            marker=markers[cat],
            markersize=marker_sizes[cat],
            linestyle=lines[cat],
            color=colors[cat],
            label=cat
        )

    if legendary:
        ax.legend(markerscale=4, title=legend_title)

    return ax


def plot3d(x, y, z, categories,
           cats=None,
           ax_pos=None, ax_labels=None,
           title=None, caption=None,
           markers='o', marker_sizes=1, lines='None',
           legendary=True, legend_title='None',
           palette='default'):
    r"""This will plot a three dimensional graph as desired.
    It will first retrieve each category inside categories,
    if cats is specified, the cats will be used instead.
    This plot function will plot each point in order of different
    categories.

    Parameters
    ----------
    x : array_like, shape (n_samples)
        `x` coordinate values for all points.
    y : array_like, shape (n_samples)
        `y` coordinate values for all points.
    z : array_like, shape (n_samples)
        `z` coordinate values for all points.
    categories : array_like, shape (n_samples)
        Categories for all points.
    cats : list, optional
        The cats used to plot, this can make plotting in order
        of different categories in order defined in cats.
    ax_pos : tuple, optional
        The positions to plot graph, provided if subplot is desired.
    ax_labels : list, optional
        The coordinate labels used in initialized axes.
        When specified, it has to be a list contained of three names,
        for `x`, `y` and `z` coordinate respectfully.
    title : str, optional
        The title of plot axes.
    caption : str, optional
        The caption of plot axes.
    markers : str, optional
        The marker used to plot the points.
    marker_sizes : int, optional
        The size of maker used to plot the points.
    lines : str, optional
        The line style used to connect points in the same category.
        If lines is 'None', no line will be plotted.
    legendary : bool, optional
        Whether plot legend information.
    legend_title : str, optional
        The legend title used to specify legend,
        when legendary is True.
    palette : str, optional
        The palette name used to assign difference color to
        different category.

    Returns
    -------
    ax : matplotlib.axes
        The axes of plotting.
    """

    x = _check_ndarray(x)
    y = _check_ndarray(y)
    z = _check_ndarray(z)
    categories = _check_ndarray(categories)

    if ax_pos is not None:
        ax = plt.subplot(*ax_pos, projection='3d')
    else:
        ax = plt.subplot(*(1, 1, 1), projection='3d')

    if ax_labels is not None:

        if len(ax_labels) != 3:
            raise ValueError(
                f'Invalid ax_labels, it needs three elements inside '
                f'for x and y axis. Current ax_labels have {len(ax_labels)} '
                f'elements.'
            )

        ax.set_xlabel(ax_labels[0], fontdict={'size': 10})
        ax.set_ylabel(ax_labels[1], fontdict={'size': 10})
        ax.set_zlabel(ax_labels[2], fontdict={'size': 10})

    if title:
        ax.set_title(label=title, fontdict={'size': 12})

    if caption:
        ax.text(
            s='\n' + caption,
            horizontalalignment='left',
            x=0, y=-0.075, z=0,
            transform=ax.transAxes,
            fontdict={'size': 8}
        )

    if cats is None:
        cats, indices = np.unique(categories, return_index=True)
        cats = cats[np.argsort(indices)]

    if not isinstance(markers, dict):
        markers = {cat: markers for cat in cats}
    if not isinstance(marker_sizes, dict):
        marker_sizes = {cat: marker_sizes for cat in cats}
    if not isinstance(lines, dict):
        lines = {cat: lines for cat in cats}

    palette = load_palette(palette)
    if np.issubdtype(cats.dtype, np.integer):
        colors = {
            cat: palette[cat % len(palette)]
            for cat in cats
        }
    else:
        colors = {
            cat: palette[idx % len(palette)]
            for idx, cat in enumerate(cats)
        }

    for cat in cats:
        ax.plot(
            x[categories == cat],
            y[categories == cat],
            z[categories == cat],
            zdir='z',
            marker=markers[cat],
            markersize=marker_sizes[cat],
            linestyle=lines[cat],
            color=colors[cat],
            label=cat
        )

    if legendary:
        ax.legend(markerscale=4, title=legend_title)

    return ax


def plotnx(adj_mat, categories,
           cats=None, G=None,
           ax_pos=None,
           title=None, caption=None,
           pos=None, with_node_labels=False,
           arrows=False, arrowstyle='-|>', arrowsize=10,
           edge_width=0.1, edge_attr=None, edge_font_size=5,
           with_edge_labels=False,
           markers='o', marker_sizes=1,
           legendary=True, legend_title='None',
           palette='default'):
    r"""This will plot a network graph as desired.
    It will first retrieve each category inside categories,
    if cats is specified, the cats will be used instead.

    Parameters
    ----------
    adj_mat : array_like, scipy.sparse.spmatrix, shape (n_nodes, n_nodes)
        The adjacency matrix of a graph.
    categories : array_like, shape (n_nodes)
        Categories for all nodes.
    cats : list, optional
        The cats used to plot, this can make assign different node size
        and node color in order.
    ax_pos : tuple, optional
        The positions to plot graph, provided if subplot is desired.
    G : networkx.Graph, optional
        The networkx.Graph to provide when adj_mat is not available.
    title : str, optional
        The title of plot axes.
    caption : str, optional
        The caption of plot axes.
    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
    with_node_labels : bool, optional
        When set to True, nodes labels will be plotted.
    arrows : bool, optional
        For directed graph, if True draw arrow heads.
    arrowstyle : str, optional
        For directed graph, choose the style of arrow heads.
    arrowsize : int, optional
        For directed graph, choose the size of the arrow heads.
    edge_width : float, optional
        The line width of edges.
    edge_attr : str, optional
        The edge attribute to be plotted on edges.
    edge_font_size : int, optional
        The size of edge attribute to be plotted on edges.
    with_edge_labels : bool, optional
        When set to True, edge labels will be plotted.
    markers : str, optional
        The shape of the nodes.
    marker_sizes : int or array_like, shape (n_nodes) , optional
        The size of the nodes, it can be a fixed integer or a list to specify
        every node.
    palette : str, optional
        The palette name used to assign difference color to
        different category.
    legendary : bool, optional
        Whether plot legend information.
    legend_title : str, optional
        The legend title used to specify legend,
        when legendary is True.

    Returns
    -------
    ax : matplotlib.axes
        The axes of plotting.

    Raises
    ------
    ValueError
        When adj_mat is not one of np.ndarray or scipy.sparse.spmatrix,
        and meanwhile G is not provided, a ValueError is raised.
    """

    if ax_pos is not None:
        ax = plt.subplot(*ax_pos)
    else:
        ax = plt.subplot(*(1, 1, 1))

    if title:
        ax.set_title(label=title, fontdict={'size': 12})

    if caption:
        ax.text(
            s='\n' + caption,
            horizontalalignment='left',
            x=0, y=-0.075,
            transform=ax.transAxes,
            fontdict={'size': 8}
        )

    if isinstance(adj_mat, np.ndarray):
        G = nx.from_numpy_matrix(adj_mat)
    elif sparse.isspmatrix(adj_mat):
        G = nx.from_scipy_sparse_array(adj_mat)
    else:
        if G is None:
            raise ValueError(
                f'Currently adj_mat should be either np.ndarray '
                f'or scipy.sparse.spmatrix, your type is {type(adj_mat)}. '
                f'You can also directly input a networkx graph '
                f'using G parameter.'
            )

    if cats is None:
        cats, indices = np.unique(categories, return_index=True)
        cats = cats[np.argsort(indices)]

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    if not isinstance(markers, dict):
        markers = {cat: markers for cat in cats}
    if not isinstance(marker_sizes, dict):
        marker_sizes = {cat: marker_sizes for cat in cats}

    palette = load_palette(palette)
    if np.issubdtype(cats.dtype, np.integer):
        colors = {
            cat: palette[cat % len(palette)]
            for cat in cats
        }
    else:
        colors = {
            cat: palette[idx % len(palette)]
            for idx, cat in enumerate(cats)
        }

    node_color = np.array([(0.0, 0.0, 0.0)] * len(G))
    for cat in cats:
        node_color[categories == cat] = colors[cat]

    for cat in cats:
        nx.draw_networkx(
            G,
            nodelist=np.array(G.nodes())[categories == cat],
            pos=pos, with_labels=with_node_labels,
            arrows=arrows, arrowstyle=arrowstyle, arrowsize=arrowsize,
            width=edge_width,
            node_shape=markers[cat],
            node_size=marker_sizes[cat],
            node_color=node_color[categories == cat],
            label=cat
        )

    if with_edge_labels:
        if edge_attr is None:
            nx.draw_networkx_edge_labels(
                G,
                pos=pos,
                font_size=edge_font_size
            )
        else:
            nx.draw_networkx_edge_labels(
                G,
                pos=pos,
                edge_labels=nx.get_edge_attributes(G, edge_attr),
                font_size=edge_font_size
            )

    if legendary:
        ax.legend(markerscale=4, title=legend_title)

    # draw_networkx will remove ticks, here we add them back
    ax.tick_params(labelleft=True, labelbottom=True)

    return ax
