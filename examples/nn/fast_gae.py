import vectorlab
import torch_geometric

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

dataset = torch_geometric.datasets.KarateClub()
transform = torch_geometric.transforms.RandomLinkSplit(
    is_undirected=True,
    split_labels=True
)
train_data, val_data, test_data = transform(dataset.data)

net = vectorlab.nn.models.FastGAE(
    encoder=torch_geometric.nn.models.GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=16,
        num_layers=3
    ),
    decoder=vectorlab.nn.models.InnerProducetDecoder(sigmoid=True)
)

explorer = vectorlab.nn.Explorer(
    net, net.loss,
    train_loader_fn='node_dataloader',
    batch_input='data',
    net_input='data.x, data.pos_edge_label_index',
    loss_input='data.pos_edge_label_index, data.neg_edge_label_index',
    num_epochs=200,
    learning_rate=1e-2,
    earlystopping_fn=None,
    writer=False,
    device='cpu'
)
explorer.train([train_data], verbose=2, save_best=False, save_last=False)

z = explorer.latent([train_data])
y = dataset.data.y.numpy()

clf = LinearSVC().fit(z, y)
yhat = clf.predict(z)
p, r, f, _ = precision_recall_fscore_support(y, yhat)

print(f'Precision: {p}, Recall: {r}, F-score: {f}')
