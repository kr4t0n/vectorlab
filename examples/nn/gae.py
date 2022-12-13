import vectorlab
import torch_geometric

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

dataset = torch_geometric.datasets.KarateClub()
torch_geometric.utils.train_test_split_edges(dataset.data)

net = vectorlab.nn.models.GAE(
    encoder=torch_geometric.nn.models.GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=16,
        num_layers=3
    ),
    decoder=vectorlab.nn.models.InnerProducetDecoder(sigmoid=True)
)
loss_fn = vectorlab.nn.GraphReconLoss()

explorer = vectorlab.nn.Explorer(
    net, loss_fn,
    train_loader_fn='node_dataloader',
    batch_input='data',
    net_input='data.x, data.train_pos_edge_index',
    loss_input='data.train_pos_edge_index',
    num_epochs=200,
    learning_rate=1e-2,
    earlystopping_fn=None,
    writer=False,
    device='cpu'
)
explorer.train(dataset, verbose=2, save_best=False, save_last=False)

z = explorer.latent(dataset)
y = dataset.data.y.numpy()

clf = LinearSVC().fit(z, y)
yhat = clf.predict(z)
p, r, f, _ = precision_recall_fscore_support(y, yhat)

print(f'Precision: {p}, Recall: {r}, F-score: {f}')
