import torch
import vectorlab

x = torch.rand(100, 3)
dataset = torch.utils.data.dataset.TensorDataset(x)

seq = torch.rand(100, 10, 3)
seq_dataset = vectorlab.data.dataset.SequenceDataset(seq, seq)

# ============= AE ==============
ae_net = vectorlab.nn.models.AE(
    encoder=vectorlab.nn.models.MLPEncoder(
        in_dims=3, hidden_dims=8, num_layers=2
    ),
    decoder=vectorlab.nn.models.MLPDecoder(
        hidden_dims=8, out_dims=3, num_layers=2,
        sigmoid=True
    )
)
ae_loss_fn = torch.nn.MSELoss()
explorer = vectorlab.nn.Explorer(
    ae_net, ae_loss_fn,
    batch_input='X', net_input='X', loss_input='X',
    num_epochs=10
)
explorer.train(dataset, verbose=2, save_last=False)

# ============= VAE ==============
vae_net = vectorlab.nn.models.VAE(
    encoder=vectorlab.nn.models.MLPVarEncoder(
        in_dims=3, hidden_dims=8, num_layers=2
    ),
    decoder=vectorlab.nn.models.MLPDecoder(
        hidden_dims=8, out_dims=3, num_layers=2,
        sigmoid=True
    )
)
explorer = vectorlab.nn.Explorer(
    vae_net, vae_net.loss_fn,
    batch_input='X', net_input='X', loss_input='X',
    num_epochs=10
)
explorer.train(dataset, verbose=2, save_last=False)

# ============= RNN AE ==============
rnn_net = vectorlab.nn.models.RNNAE(
    encoder=vectorlab.nn.models.GRUEncoder(
        in_dims=3, hidden_dims=8, num_layers=2
    ),
    decoder=vectorlab.nn.models.GRUDecoder(
        in_dims=3, hidden_dims=8, out_dims=3, num_layers=2,
        sigmoid=True
    )
)
gru_loss_fn = torch.nn.MSELoss()
explorer = vectorlab.nn.Explorer(
    rnn_net, gru_loss_fn,
    train_loader_fn='pad_seqs_dataloader',
    batch_input='X, X_lens, y, y_lens', net_input='X', loss_input='y',
    num_epochs=10
)
explorer.train(seq_dataset, verbose=2, save_last=False)
