import torch
import vectorlab

x = torch.randint(0, 10, (10, 4))
y = torch.randint(0, 20, (10, 4))

dataset = vectorlab.data.dataset.SequenceDataset(x, y)

net = vectorlab.nn.models.Seq2Seq(
    encoder_embedding=torch.nn.Embedding(10, 3),
    encoder=vectorlab.nn.models.GRUEncoder(
        in_dims=3, hidden_dims=8, num_layers=2
    ),
    decoder_embedding=torch.nn.Embedding(20, 3),
    decoder=vectorlab.nn.models.GRUDecoder(
        in_dims=3, hidden_dims=8, out_dims=20, num_layers=2,
        sigmoid=False
    ),
    start_token=0,
    teacher_forcing=.0
)
loss_fn = vectorlab.nn.SequenceNLLLoss()
explorer = vectorlab.nn.Explorer(
    net, loss_fn,
    train_loader_fn='pad_seqs_dataloader',
    train_loader_kwargs={'batch_first': True},
    batch_input='X, X_lens, y, y_lens', net_input='X, y', loss_input='y',
    num_epochs=10
)
explorer.train(dataset, verbose=2, save_last=False)
