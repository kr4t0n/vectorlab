import torch
import pytest

from vectorlab.nn import Explorer


@pytest.mark.parametrize('earlystopping_fn', [None, 'desc_es'])
def test_explorer(earlystopping_fn):

    X = torch.rand(10, 3)
    y = torch.rand(10, 2)
    dataset = torch.utils.data.dataset.TensorDataset(*(X, y))

    net = torch.nn.Linear(3, 2)
    loss_fn = torch.nn.MSELoss()

    explorer = Explorer(
        net, loss_fn,
        batch_input='X, y', net_input='X', loss_input='y',
        k=2,
        num_workers=0, num_epochs=10,
        earlystopping_fn=earlystopping_fn
    )

    explorer.train(dataset, verbose=1, save_last=False)
    explorer.k_fold_train(dataset, verbose=1)
    explorer.inference(dataset, verbose=1)
    explorer.get_summary(dataset, verbose=1)
