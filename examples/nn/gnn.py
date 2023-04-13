import torch
import vectorlab
import torch_geometric

dataset = torch_geometric.datasets.KarateClub()

net = vectorlab.nn.models.GClassifier(
    encoder=torch_geometric.nn.models.GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=16,
        num_layers=3
    ),
    classifier=vectorlab.nn.models.MLP(
        [16, 8, 4, dataset.num_classes]
    )
)
loss_fn = torch.nn.CrossEntropyLoss()

explorer = vectorlab.nn.Explorer(
    net, loss_fn,
    train_loader_fn='node_dataloader',
    batch_input='data',
    net_input='data.x, data.edge_index', loss_input='data.y',
    num_epochs=200,
    earlystopping_fn=None,
    writer=False,
    device='cpu'
)
explorer.train(dataset, verbose=2, save_best=False, save_last=False)
