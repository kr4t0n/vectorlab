from re import A
import numpy as np
from vectorlab import graph

from vectorlab.graph import graph_er_generator
from vectorlab.plot import init_plot, plot2d, plot3d, plotnx

n_samples = 100
n_cats = 4

x = np.random.rand(n_samples)
y = np.random.rand(n_samples)
z = np.random.rand(n_samples)
categories = np.random.randint(0, n_cats, n_samples)

n_nodes = 10
n_node_cats = 4

adj_mat = graph_er_generator(n_nodes)
node_categories = np.random.rand(0, n_node_cats, n_nodes)


def test_plot2d():
    init_plot()
    plot2d(x=x, y=y, categories=categories)


def test_plot3d():
    init_plot()
    plot3d(x=x, y=y, z=z, categories=categories)


def test_plotnx():
    init_plot()
    plotnx(adj_mat=adj_mat, categories=node_categories)
