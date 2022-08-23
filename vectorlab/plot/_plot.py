"""
The functions are focused on plotting relationship between variables.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ._palettes import load_palette


def init_plot(width=10, height=8, dpi=100,
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

    n, e, w, s = (0.95, 0.9, 0.1, 0.15)
    fig.subplots_adjust(bottom=s, right=e, left=w, top=n)

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

    if ax_pos is not None:
        ax = plt.subplot(*ax_pos)
    else:
        ax = plt.subplot(*(1, 1, 1))

    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0], fontdict={'size': 10})
        ax.set_ylabel(ax_labels[1], fontdict={'size': 10})

    if title:
        ax.set_title(label=title, fontdict={'size': 12})

    if caption:
        ax.text(
            s='\n' + caption,
            horizontalalignment='left',
            x=0, y=-0.128,
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
        plt.legend(markerscale=4, title=legend_title)

    return ax
