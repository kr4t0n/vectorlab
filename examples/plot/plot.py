import numpy as np
import vectorlab as vl

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
categories = np.random.randint(0, 2, 100)
adj_mat = vl.graph.graph_er_generator(10, 0.5)
node_categories = np.random.randint(0, 2, 10)


fig = vl.plot.init_plot(
    width=30, height=8,
    ax_labels=('Sup X', 'Sup Y'),
    title='Sample Title',
    style='whitegrid',
    despine=False
)

vl.plot.plot2d(
    x=x, y=y, categories=categories,
    ax_pos=(1, 3, 1), ax_labels=('x', 'y'),
    title='Sample title 2d',
    caption='Sample caption 2d',
    palette='high_contrast',
    legendary=True, legend_title='Legend 2d'
)

vl.plot.plot3d(
    x=x, y=y, z=z, categories=categories,
    ax_pos=(1, 3, 2), ax_labels=('x', 'y', 'z'),
    title='Sample title 3d',
    caption='Sample caption 3d',
    palette='fifth_dimension',
    legendary=True, legend_title='Legend 3d'
)

vl.plot.plotnx(
    adj_mat=adj_mat, categories=node_categories,
    ax_pos=(1, 3, 3),
    title='Sample title nx',
    caption='Sample caption nx',
    palette='long_contrast',
    legendary=True, legend_title='Legend nx',
    marker_sizes=5
)

vl.plot.show_plot()
