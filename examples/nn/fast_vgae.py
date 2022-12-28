import vectorlab
import numpy as np
import torch_geometric

from scipy import sparse

dataset = torch_geometric.datasets.KarateClub()
transform = torch_geometric.transforms.RandomLinkSplit(
    is_undirected=True,
    split_labels=True
)
train_data, val_data, test_data = transform(dataset.data)

net = vectorlab.nn.models.FastVGAE(
    encoder=vectorlab.nn.models.VarGCNEncoder(
        in_dims=dataset.num_node_features,
        hidden_dims=16,
        num_layers=3,
        dropout=.0
    ),
    decoder=vectorlab.nn.models.InnerProducetDecoder(sigmoid=True)
)

explorer = vectorlab.nn.Explorer(
    net, net.loss_fn,
    train_loader_fn='node_dataloader',
    batch_input='data',
    net_input='data.x, data.pos_edge_label_index',
    loss_input='data.pos_edge_label_index, data.neg_edge_label_index',
    num_epochs=1000,
    learning_rate=1e-2,
    earlystopping_fn=None,
    writer=False,
    device='cpu'
)
explorer.train([train_data], verbose=2, save_best=False, save_last=False)


# inference process

adj = explorer.inference([train_data])
row, col = np.indices(adj.shape)
row = row.reshape(-1)
col = col.reshape(-1)
adj_mat = sparse.coo_matrix(
    (
        adj.reshape(-1),
        (row, col)
    )
)
adj_mat = vectorlab.graph.to_deterministic_graph(
    adj_mat, thres=0.9999,
    ensure_connective=True
)
adj_mat = vectorlab.graph.make_connectivity(adj_mat)
adj_mat = vectorlab.graph.remove_self_loop(adj_mat)


# plot result

G = torch_geometric.utils.to_networkx(dataset.data, to_undirected=True)

vectorlab.plot.init_plot()

vectorlab.plot.plotnx(
    adj_mat=None,
    G=G,
    categories=np.zeros(G.number_of_nodes()),
    ax_pos=(1, 2, 1)
)

vectorlab.plot.plotnx(
    adj_mat=adj_mat,
    categories=np.zeros(adj_mat.shape[0]),
    ax_pos=(1, 2, 2)
)


vectorlab.plot.show_plot()
