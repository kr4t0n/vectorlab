import time
import numpy as np

from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import torch

from torchinfo import summary
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch

from ..base import SLMixin, Accumulator
from ._earlystopping import EarlyStopping
from ..data.dataloader._torch_dataloader import (
    PadSeqDataLoader, PadSeqsDataLoader
)
from ..data.dataloader._torch_geometric_dataloader import (
    NodeDataLoader,
    GraphDataLoader, MaskGraphDataLoader
)


class Explorer(SLMixin):
    r"""The Explorer class to train a neural network.

    The Explorer helps to train a model with a certain set of arguments.
    The Explorer first initialized with a pre-set of arguments. And mainly,
    there are several methods implemented inside. Normal training helps to
    train the model in a normal way, while the k fold training tries to train
    the model in cross validation manner. Also a evaluation method is used to
    criticize the trained model, and predict methods helps with the inference
    phase. One reset method is used to reset all parameters and states inside.

    Parameters
    ----------
    net : torch.nn.Module
        The defined neural network.
    loss_fn : torch.nn.Module
        The loss function to criticize the performance.
    k : int
        The hyper-parameter k of k fold cross validation.
    batch_size : int
        The batch size of loading data.
    num_workers : int
        The number of subprocesses to load data.
    num_epochs : int
        The number of epoch to train the neural network.
    train_loader_fn : str, callable
        The data loader function to be used to load data from dataset
        in training process, currently supports nn and gnn.
    train_loader_kwargs: dict
        The extra arguments for initialize the training data loader.
    valid_loader_fn : str, callable
        The data loader function to be used to load data from dataset
        in validation process.
    valid_loader_kwargs : dict
        The extra arguments for initialize the validation data loader.
    test_loader_fn : str, callable
        The data loader function to be used to load data from dataset
        in test process.
    test_loader_kwargs : dict
        The extra arguments for initialize the test data loader.
    batch_input : str
        The data annotation for batch.
    net_input : str
        The data annotation from batch to feed into neural network.
    loss_input : str
        The data annotation from batch to feed into loss function.
    optimizer_fn : str, callable
        The optimizer to be used to do optimization.
    learning_rate : float
        The learning rate used to update the parameters.
    weight_decay : float
        The weight decay factor to regularize the parameters.
    optimizer_kwargs: dict
        The extra arguments for initialize the optimizer.
    scheduler_fn : str, callable
        The scheduler to be used to adjust the learning rate.
    scheduler_kwargs : dict
        The extra arguments for initialize the scheduler.
    earlystopping_fn : str, callable
        The earlystopping to be used to avoid overfitting.
    earlystopping_metric : str
        The montior metric for earlystopping.
    earlystopping_kwarg : dict
        The extra arguments for initialize the earlystopping.
    device : str, list, optional
        If given specified device or a list of devices, the explorer will use
        such device(s) as desired. If no device is specified, the explorer
        will automatically choose the device(s).
    writer : bool
        Whether to use summary writer or not.
    writer_dir : str
        The directory to store the information from summary writer.
    writer_comment : str
        The extra comment used to initialize the summary writer.
    parameters_dict : dict
        The extra parameters dictionary to specify.

    Attributes
    ----------
    net_ : torch.nn.Module
        The defined neural network.
    loss_fn_ : torch.nn.Module
        The loss function to criticize the performance.
    k_ : int
        The hyper-parameter k of k fold cross validation.
    batch_size_ : int
        The batch size of loading data.
    num_workers_ : int
        The number of subprocesses to load data.
    num_epochs_ : int
        The number of epoch to train the neural network.
    _loader_fns_ : dict
        The dictionary stored different data loader functions.
    loader_fn_ : torch.utils.data.DataLoader, torch_geometric.loader.DataLoader
        The data loader to load the data from dataset.
    batch_input_ : list
        The list of data annotations for batch.
    net_input_ : list
        The list of data annotations from batch to feed into neural network.
    loss_input_ : list
        The list of data annotations from batch to feed into loss function.
    batch_input_dict_ : dict
        The dictionary of data annotation and its index in batch data.
    net_input_ind_ : tuple
        The tuple of net_input in indices.
    loss_input_ind_ : tuple
        The tuple of loss_input in indices.
    _optimizers_ : dict
        The dictionary stored different optimizers.
    optimizer_fn_ : torch.optim.Optimizer
        The given callable function to initialize corresponding optimizer.
    learning_rate_ : float
        The learning rate used to update the parameters.
    weight_decay_ : float
        The weight decay factor to regularize the parameters.
    optimizer_kwargs_: dict
        The extra arguments for initialize the optimizer.
    optimizer_ : torch.optim.Optimizer
        The optimizer used to train the model.
    _schedulers_ : dict
        The dictionary stored different schedulers.
    scheduler_fn_ : torch.optim.lr_scheduler
        The given callable function to initialize corresponding scheduler.
    scheduler_kwargs_ : dict
        The extra arguments for initialize the scheduler.
    scheduler_ : torch.optim.lr_scheduler
        The scheduler to decay the learning rate of optimizer.
    _earlystoppings_ : dict
        The dictionary stored different earlystoppings.
    earlystopping_fn_ : callabe
        The earlystopping to be used to avoid overfitting.
    earlystopping_metric_ : str
        The montior metric for earlystopping.
    earlystopping_kwarg_ : dict
        The extra arguments for initialize the earlystopping.
    earlystopping_ : callable
        The earlystopping to break epoch iteration.
    device_ : torch.device
        The first device used to store the neural network and data.
    devices_ : list
        The list of potential devices that would be used.
    writer_ : bool
        If use summary writer of tensorboard to log the process.
    train_writer_ : torch.utils.tensorboard.SummaryWriter
        The summary writer for training process.
    valid_writer_ : torch.utils.tensorboard.SummaryWriter
        The summary writer for validation process.
    writer_dir_ : str
        The log directory for summary writer.
    writer_comment : str
        The comment used to identify the summary writer.
    parameters_dict_ : dict
        The dictionary stored the parameters that initialized the
        explorer.
    train_loss_ : float
        The loss criterion of trained neural network over the training
        dataset.
    valid_loss_ : float
        The loss criterion of trained neural network over the validation
        dataset.
    best_loss_ : float
        The best loss criterion of trained neural network over training
        dataset if validation dataset is not provided, over validation
        dataset otherwise.
    k_train_loss_ : float
        The loss criterion of trained neural network over the training
        dataset using k fold cross validation.
    k_valid_loss_ : float
        The loss criterion of trained neural network over the validation
        dataset using k fold cross validation.
    outputs_ : np.ndarray
        The predicting results of test dataset using the trained neural
        network.
    """

    _loader_fns_ = {
        'nn': torch.utils.data.DataLoader,
        'pad_seq': PadSeqDataLoader,
        'pad_seqs': PadSeqsDataLoader,
        'gnn_node': NodeDataLoader,
        'gnn_graph': GraphDataLoader,
        'gnn_zero_mask': lambda *args, **kwargs: MaskGraphDataLoader(
            *args,
            mask_method='zeros',
            **kwargs
        ),
        'gnn_random_mask': lambda *args, **kwargs: MaskGraphDataLoader(
            *args,
            mask_method='random',
            **kwargs
        )
    }

    _optimizers_ = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD
    }

    _schedulers_ = {
        'step_lr': torch.optim.lr_scheduler.StepLR,
        'cosine_lr': torch.optim.lr_scheduler.CosineAnnealingLR
    }

    _earlystoppings_ = {
        'earlystopping': EarlyStopping
    }

    def __init__(self,
                 net, loss_fn,
                 k=5,
                 batch_size=32, num_workers=8, num_epochs=100,
                 train_loader_fn='nn', train_loader_kwargs=None,
                 valid_loader_fn=None, valid_loader_kwargs=None,
                 test_loader_fn=None, test_loader_kwargs=None,
                 batch_input=None, net_input=None, loss_input=None,
                 optimizer_fn='adamw',
                 learning_rate=0.1, weight_decay=0, optimizer_kwargs=None,
                 scheduler_fn='cosine_lr',
                 scheduler_kwargs=None,
                 earlystopping_fn='earlystopping',
                 earlystopping_metric='loss', earlystopping_kwargs=None,
                 device=None,
                 writer=False, writer_dir=None, writer_comment='',
                 parameters_dict=None):

        super().__init__()

        self._init_devices_(device)

        self.k_ = int(k)
        self.batch_size_ = int(batch_size)
        self.num_workers_ = int(num_workers)
        self.num_epochs_ = int(num_epochs)

        self.train_loader_fn_ = self._init_loader_fn(train_loader_fn)
        self.train_loader_kwargs_ = \
            train_loader_kwargs if train_loader_kwargs else {}

        if valid_loader_fn is None:
            self.valid_loader_fn_ = self.train_loader_fn_
        else:
            self.valid_loader_fn_ = self._init_loader_fn(valid_loader_fn)
        self.valid_loader_kwargs_ = \
            valid_loader_kwargs if valid_loader_kwargs else {}

        if test_loader_fn is None:
            self.test_loader_fn_ = self.train_loader_fn_
        else:
            self.test_loader_fn_ = self._init_loader_fn(test_loader_fn)
        self.test_loader_kwargs_ = \
            test_loader_kwargs if test_loader_kwargs else {}

        self.net_ = net
        self.loss_fn_ = loss_fn

        self.batch_input_ = batch_input
        self.net_input_ = net_input
        self.loss_input_ = loss_input
        self._init_input_format()

        if len(self.devices_) > 1:
            self.net_ = torch.nn.DataParallel(
                self.net_, device_ids=self.devices_
            )

        self.optimizer_fn_ = optimizer_fn
        self.learning_rate_ = learning_rate
        self.weight_decay_ = weight_decay
        self.optimizer_kwargs_ = optimizer_kwargs
        self._init_optimizer()

        self.scheduler_fn_ = scheduler_fn
        self.scheduler_kwargs_ = scheduler_kwargs
        self._init_scheduler()

        self.earlystopping_fn_ = earlystopping_fn
        self.earlystopping_kwargs_ = earlystopping_kwargs
        self.earlystopping_metric_ = earlystopping_metric
        self._init_earlystopping()

        self.writer_ = writer
        self.writer_dir_ = writer_dir
        self.writer_comment_ = writer_comment

        self._init_parameters_dict_(parameters_dict)

        return

    def _init_devices_(self, device=None):
        r"""Automatically initialize the devices used to store the data
        and model.

        If certain device or a list of devices are given, such device(s)
        will be used to store the data and model, otherwise, all available
        devices will be used.

        Parameters
        ----------
        device : str, list, optional
            If given specified device or a list of devices, the explorer
            will use such device(s) as desired. If no device is specified,
            the explorer will automatically choose the device(s).

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if device is None:
            if torch.cuda.is_available():
                self.devices_ = [
                    torch.device(f'cuda:{i}')
                    for i in range(torch.cuda.device_count())
                ]
                self.device_ = self.devices_[0]
            elif (
                hasattr(torch.backends, 'mps')
            ) and torch.backends.mps.is_available():
                self.devices_ = [torch.device('mps')]
                self.device_ = self.devices_[0]
            else:
                self.devices_ = [torch.device('cpu')]
                self.device_ = self.devices_[0]
        else:
            if isinstance(device, list):
                self.devices_ = [torch.device(_) for _ in device]
                self.device_ = self.devices_[0]
            else:
                self.devices_ = [torch.device(device)]
                self.device_ = self.devices_[0]

        return self

    def _init_loader_fn(self, loader_fn):
        r"""Generate corresponding data loader functions.

        Parameters
        ----------
        loader_fn : str, callable
            A callable data loader function or a string to generate
            pre-defined data loader function.

        Returns
        -------
        loader_fn : callable
            A callable data loader function.
        """

        if callable(loader_fn):
            return loader_fn
        elif isinstance(loader_fn, str):
            return self._loader_fns_[loader_fn]

    def _init_optimizer(self):
        r"""Initialize proper optimizer

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.optimizer_kwargs_ = \
            self.optimizer_kwargs_ if self.optimizer_kwargs_ else {}

        # update learning rate and weight decay
        self.optimizer_kwargs_.update(
            {
                'lr': self.learning_rate_,
                'weight_decay': self.weight_decay_
            }
        )

        if not callable(self.optimizer_fn_):
            self.optimizer_fn_ = self._optimizers_[self.optimizer_fn_]

        self.optimizer_ = self.optimizer_fn_(
            self.net_.parameters(),
            **self.optimizer_kwargs_
        )

        return self

    def _init_scheduler(self):
        r"""Initialize proper scheduler

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if self.scheduler_fn_ is None:
            self.scheduler_kwargs_ = \
                self.scheduler_kwargs_ if self.scheduler_kwargs_ else {}
            self.scheduler_ = None
        else:
            if callable(self.scheduler_fn_):
                self.scheduler_kwargs_ = \
                    self.scheduler_kwargs_ if self.scheduler_kwargs_ else {}
            else:
                if self.scheduler_kwargs_ is None:
                    if self.scheduler_fn_ == 'step_lr':
                        self.scheduler_kwargs_ = {
                            'step_size': self.num_epochs_ // 4
                        }
                    elif self.scheduler_fn_ == 'cosine_lr':
                        self.scheduler_kwargs_ = {
                            'T_max': self.num_epochs_
                        }
                    else:
                        self.scheduler_kwargs_ = {}
                else:
                    self.scheduler_kwargs_ = self.scheduler_kwargs_

                self.scheduler_fn_ = self._schedulers_[self.scheduler_fn_]

            self.scheduler_ = self.scheduler_fn_(
                self.optimizer_,
                **self.scheduler_kwargs_
            )

        return self

    def _init_earlystopping(self):
        r"""Initialize proper earlystopping

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.earlystopping_kwargs_ = \
            self.earlystopping_kwargs_ if self.earlystopping_kwargs_ else {}

        if self.earlystopping_fn_ is None:
            self.earlystopping_ = None
        else:
            if not callable(self.earlystopping_fn_):
                self.earlystopping_fn_ = \
                    self._earlystoppings_[self.earlystopping_fn_]

            self.earlystopping_ = self.earlystopping_fn_(
                **self.earlystopping_kwargs_
            )

        return self

    def _init_input_format(self):
        r"""Automatically analysis the data input format.

        Firstly, analysis the batch data input format, and generate
        corresponding data indices for net input and loss input.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.batch_input_ = self.batch_input_.replace(' ', '').split(',')
        self.net_input_ = self.net_input_.replace(' ', '').split(',')

        if self.loss_input_ is not None:
            self.loss_input_ = self.loss_input_.replace(' ', '').split(',')

        self.batch_input_dict_ = \
            {k: idx for idx, k in enumerate(self.batch_input_)}

        self.net_input_ind_ = tuple(
            (
                self.batch_input_dict_[k.split('.')[0]],
                *k.split('.')[1:]
            )
            for k in self.net_input_
        )

        if self.loss_input_ is not None:
            self.loss_input_ind_ = tuple(
                (
                    self.batch_input_dict_[k.split('.')[0]],
                    *k.split('.')[1:]
                )
                for k in self.loss_input_
            )
        else:
            self.loss_input_ind_ = tuple()

        return self

    def _init_parameters_dict_(self, parameters_dict):
        r"""Generate the parameters dictionary that initialized
        the explorer.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.parameters_dict_ = {
            'batch_size': self.batch_size_,
            'num_workers': self.num_workers_,
            'num_epochs': self.num_epochs_,
        }

        if self.optimizer_kwargs_:
            self.parameters_dict_.update(self.optimizer_kwargs_)
        if self.scheduler_kwargs_:
            self.parameters_dict_.update(self.scheduler_kwargs_)
        if self.earlystopping_kwargs_:
            self.parameters_dict_.update(self.earlystopping_kwargs_)

        if parameters_dict is not None:
            self.parameters_dict_.update(parameters_dict)

        return self

    def reset(self):
        r"""The reset method.

        This method will reset the parameters inside the neural network
        and re-initialized the optimizer and scheduler.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if len(self.devices_) > 1:
            self.net_.module.reset_parameters()
        else:
            self.net_.reset_parameters()

        self.optimizer_ = self.optimizer_fn_(
            self.net_.parameters(),
            **self.optimizer_kwargs_
        )
        if self.scheduler_fn_:
            self.scheduler_ = self.scheduler_fn_(
                self.optimizer_,
                **self.scheduler_kwargs_
            )
        if self.earlystopping_fn_:
            self.earlystopping_ = self.earlystopping_fn_(
                **self.earlystopping_kwargs_
            )

        return self

    def _train(self, loader):
        """This is the abstract train method.

        This is the abstract train method that should be implemented in
        children class, and it will be used in the training process.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing training data to train the neural
            network.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.net_.train()

        for batch in loader:

            if not isinstance(batch, (tuple, list)):
                batch = (batch, )

            net_input = tuple(
                batch[ind[0]].to(self.device_)
                if len(ind) == 1
                else getattr(batch[ind[0]], ind[1]).to(self.device_)
                for ind in self.net_input_ind_
            )
            loss_input = tuple(
                batch[ind[0]].to(self.device_)
                if len(ind) == 1
                else getattr(batch[ind[0]], ind[1]).to(self.device_)
                for ind in self.loss_input_ind_
            )

            self.optimizer_.zero_grad()

            output = self.net_(*net_input)
            output = output if isinstance(output, tuple) else (output, )

            loss = self.loss_fn_(
                *output,
                *loss_input
            )
            loss.backward()
            self.optimizer_.step()

        if self.scheduler_:
            self.scheduler_.step()
        if self.earlystopping_:
            self.earlystopping_.step()

        return self

    def _loss_evaluate(self, loader):
        r"""The loss metrics evaluation method.

        We calculate the overall loss as our evaluation metrics, while the
        way of calculating the loss is same as the process in the train part.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing data to evaluate the trained neural
            network.

        Returns
        -------
        metric : dict
            The dictionary contained the evaluation metrics.
        """

        self.net_.eval()

        accumulator = Accumulator(2, ['loss', 'n_samples'])

        with torch.no_grad():

            for batch in loader:

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                if isinstance(batch[0], torch.Tensor):
                    n_samples = batch[0].shape[0]
                elif isinstance(batch[0], Data):
                    n_samples = batch[0].num_nodes
                elif isinstance(batch[0], Batch):
                    n_samples = batch[0].num_graphs

                net_input = tuple(
                    batch[ind[0]].to(self.device_)
                    if len(ind) == 1
                    else getattr(batch[ind[0]], ind[1]).to(self.device_)
                    for ind in self.net_input_ind_
                )
                loss_input = tuple(
                    batch[ind[0]].to(self.device_)
                    if len(ind) == 1
                    else getattr(batch[ind[0]], ind[1]).to(self.device_)
                    for ind in self.loss_input_ind_
                )

                output = self.net_(*net_input)
                output = output if isinstance(output, tuple) else (output, )

                loss = self.loss_fn_(
                    *output,
                    *loss_input
                ).item()
                accumulator.add(
                    {
                        'loss': loss * n_samples,
                        'n_samples': n_samples
                    }
                )

        metrics = {
            'loss': accumulator.get('loss') / accumulator.get('n_samples')
        }

        return metrics

    def _loss_acc_evaluate(self, loader):
        r"""The loss and accuracy evaluation method.

        We calculate the overall and accuracy as our evaluation metrics,
        while the way of calculating the loss is same as the process in
        the train part while accuracy is often used when served as a
        classification problem.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing data to evaluate the trained neural
            network.

        Returns
        -------
        metric : dict
            The dictionary contained the evaluation metrics.
        """

        self.net_.eval()

        accumulator = Accumulator(3, ['loss', 'acc', 'n_samples'])

        with torch.no_grad():

            for batch in loader:

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                if isinstance(batch[0], torch.Tensor):
                    n_samples = batch[0].shape[0]
                elif isinstance(batch[0], Data):
                    n_samples = batch[0].num_nodes
                elif isinstance(batch[0], Batch):
                    n_samples = batch[0].num_graphs

                net_input = tuple(
                    batch[ind[0]].to(self.device_)
                    if len(ind) == 1
                    else getattr(batch[ind[0]], ind[1]).to(self.device_)
                    for ind in self.net_input_ind_
                )
                loss_input = tuple(
                    batch[ind[0]].to(self.device_)
                    if len(ind) == 1
                    else getattr(batch[ind[0]], ind[1]).to(self.device_)
                    for ind in self.loss_input_ind_
                )

                y_hat = self.net_(*net_input)

                loss = self.loss_fn_(y_hat, *loss_input).item()
                acc = (y_hat.argmax(axis=1) == loss_input[0]).sum().item()

                accumulator.add(
                    {
                        'loss': loss * n_samples,
                        'acc': acc,
                        'n_samples': n_samples
                    }
                )

        metrics = {
            'loss': accumulator.get('loss') / accumulator.get('n_samples'),
            'acc': accumulator.get('acc') / accumulator.get('n_samples')
        }

        return metrics

    def _evaluate(self, loader):
        r"""This it the abstract performance evaluation method.

        This is the abstract evaluate method that should be implemented in
        children class, that it will evaluate the performance of trained
        neural network.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.
        """

        return self._loss_evaluate(loader)

    def _inference(self, loader, verbose=False):
        r"""This is the abstract inference method.

        This is the abstract inference method that should be implemented in
        children class, that it will inference the result of inputs using
        trained neural network.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.
        verbose : bool, optional
            If show the progress of inference process.

        Returns
        -------
        outputs : np.ndarray
            The outputs inferred by trained neural network.
        """

        outputs = []

        self.net_.eval()

        with torch.no_grad():

            for batch in tqdm(loader, ascii=True, disable=not verbose):

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                net_input = tuple(
                    batch[ind[0]].to(self.device_)
                    if len(ind) == 1
                    else getattr(batch[ind[0]], ind[1]).to(self.device_)
                    for ind in self.net_input_ind_
                )

                output = self.net_(*net_input)
                outputs.append(output.detach().cpu().tolist())

        outputs = np.concatenate(outputs)

        return outputs

    def _latent(self, loader, verbose=False):
        r"""This is the abstract latent method.

        This is the abstract getting latent method that should be implemented
        in children class, that it will get latent representation from inputs
        using trained neural network sub-module, usually an encoder.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.
        verbose : bool, optional
            If show the progress of inference process.

        Returns
        -------
        outputs : np.ndarray
            The outputs inferred by trained neural network.
        """

        outputs = []

        self.net_.eval()

        with torch.no_grad():

            for batch in tqdm(loader, ascii=True, disable=not verbose):

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                net_input = tuple(
                    batch[ind[0]].to(self.device_)
                    if len(ind) == 1
                    else getattr(batch[ind[0]], ind[1]).to(self.device_)
                    for ind in self.net_input_ind_
                )

                output = self.net_.forward_latent(*net_input)
                outputs.append(output.detach().cpu().tolist())

        outputs = np.concatenate(outputs)

        return outputs

    def train(
        self,
        train_dataset, valid_dataset=None,
        save_best=False, save_last=True,
        verbose=False
    ):
        r"""Train the neural network for once.

        Train the neural network using the pre-defined arguments for once.
        It is often after using k fold training to train the neural network
        over the whole dataset. The training loss will be store inside the
        attribute train_loss, if a valid dataset is provided, the validation
        loss will also be computed and stored inside the attribute valid_loss.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            The training dataset.
        valid_dataset : torch.utils.data.Dataset, optional
            The validation dataset.
        save_best : bool, optional
            If save the model parameters with best loss.
        save_last : bool, optional
            If save the last model parameters.
        verbose : bool, optional
            If show the progress of training process.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if verbose:
            print(
                f'Training with parameters: {self.__repr_parameters__()} '
                f'on devices {self.devices_}'
            )

        self.best_loss_ = np.Inf

        if self.writer_:
            self.train_writer_ = SummaryWriter(
                log_dir=f'{self.writer_dir_}{self.writer_comment_}_train'
                        if self.writer_dir_ else self.writer_dir_,
                comment=f'{self.writer_comment_}_train'
            )

            if valid_dataset is not None:
                self.valid_writer_ = SummaryWriter(
                    log_dir=f'{self.writer_dir_}{self.writer_comment_}_valid'
                            if self.writer_dir_ else self.writer_dir_,
                    comment=f'{self.writer_comment_}_valid'
                )

        self.net_.to(self.device_)
        self.loss_fn_.to(self.device_)

        train_loader = self.train_loader_fn_(
            train_dataset,
            batch_size=self.batch_size_,
            num_workers=self.num_workers_,
            shuffle=True,
            **self.train_loader_kwargs_
        )

        if valid_dataset is not None:
            valid_loader = self.valid_loader_fn_(
                valid_dataset,
                batch_size=self.batch_size_,
                num_workers=self.num_workers_,
                shuffle=False,
                **self.valid_loader_kwargs_
            )

        for epoch in range(self.num_epochs_):
            self._train(train_loader)

            if self.writer_ or self.earlystopping_ or save_best:
                train_metrics_ = self._evaluate(train_loader)

                if valid_dataset is not None:
                    valid_metrics_ = self._evaluate(valid_loader)

            if self.earlystopping_:
                if valid_dataset is None:
                    self.earlystopping_.record_metric(
                        train_metrics_[self.earlystopping_metric_]
                    )
                else:
                    self.earlystopping_.record_metric(
                        valid_metrics_[self.earlystopping_metric_]
                    )

            if self.writer_:
                for metric, value in train_metrics_.items():
                    self.train_writer_.add_scalar(metric, value, epoch)

                if valid_dataset is not None:
                    for metric, value in valid_metrics_.items():
                        self.valid_writer_.add_scalar(metric, value, epoch)

            if save_best:
                if valid_dataset is None:
                    if train_metrics_['loss'] < self.best_loss_:
                        self.best_loss_ = train_metrics_['loss']
                        self._save_best_model()
                else:
                    if valid_metrics_['loss'] < self.best_loss_:
                        self.best_loss_ = valid_metrics_['loss']
                        self._save_best_model()

            if self.earlystopping_ and self.earlystopping_.is_done():
                break

        train_metrics_ = self._evaluate(train_loader)
        self.train_loss_ = train_metrics_['loss']

        if self.writer_:
            self.train_writer_.add_hparams(
                self.parameters_dict_,
                {
                    'hparam/' + metric: value
                    for metric, value in train_metrics_.items()
                }
            )

        if valid_dataset is not None:
            valid_metrics_ = self._evaluate(valid_loader)
            self.valid_loss_ = valid_metrics_['loss']

            if self.writer_:
                self.valid_writer_.add_hparams(
                    self.parameters_dict_,
                    {
                        'hparam/' + metric: value
                        for metric, value in valid_metrics_.items()
                    }
                )
        else:
            self.valid_loss_ = np.Inf

        if save_last:
            self._save_last_model()

        if verbose:
            print(
                f'Result: train loss {self.train_loss_:.6f}, '
                f'valid loss {self.valid_loss_:.6f}'
            )

        return self

    def k_fold_train(self, train_dataset, verbose=False):
        r"""Trained the neural network using k fold cross validation.

        Train the neural network using the pre-defined arguments with
        k fold cross validation manner. The average of training loss will
        be stored inside the attribute k_train_loss, while the average of
        validation loss will be stored inside the attribute k_valid_loss.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            The training dataset.
        verbose : bool, optional
            If show the progress of k-fold training process.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if verbose:
            print(
                f'Training with parameters: k={self.k_}, '
                f'{self.__repr_parameters__()} '
                f'on devices {self.devices_}'
            )

        accumulator = Accumulator(3, ['k_train_loss', 'k_valid_loss', 'k'])

        kf = KFold(n_splits=self.k_)
        for k, index in enumerate(kf.split(train_dataset)):

            (train_index, valid_index) = index

            if self.writer_:
                k_train_writer = SummaryWriter(
                    log_dir=f'{self.writer_dir_}{self.writer_comment_}_'
                            f'k_fold_{k}_train'
                            if self.writer_dir_ else self.writer_dir_,
                    comment=f'{self.writer_comment_}_k_fold_{k}_train'
                )
                k_valid_writer = SummaryWriter(
                    log_dir=f'{self.writer_dir_}{self.writer_comment_}_'
                            f'k_fold_{k}_valid'
                            if self.writer_dir_ else self.writer_dir_,
                    comment=f'{self.writer_comment_}_k_fold_{k}_valid'
                )

            self.reset()

            self.net_.to(self.device_)
            self.loss_fn_.to(self.device_)

            k_train_dataset = Subset(train_dataset, train_index)
            k_valid_dataset = Subset(train_dataset, valid_index)

            k_train_dataloader = self.train_loader_fn_(
                k_train_dataset,
                batch_size=self.batch_size_,
                num_workers=self.num_workers_,
                shuffle=True,
                **self.train_loader_kwargs_
            )
            k_valid_dataloader = self.valid_loader_fn_(
                k_valid_dataset,
                batch_size=self.batch_size_,
                num_workers=self.num_workers_,
                shuffle=False,
                **self.valid_loader_kwargs_
            )

            for epoch in range(self.num_epochs_):
                self._train(k_train_dataloader)

                if self.writer_ or self.earlystopping_:
                    k_train_metrics_ = self._evaluate(k_train_dataloader)
                    k_valid_metrics_ = self._evaluate(k_valid_dataloader)

                if self.earlystopping_:
                    self.earlystopping_.record_metric(
                        k_valid_metrics_[self.earlystopping_metric_]
                    )

                if self.writer_:
                    for metric, value in k_train_metrics_.items():
                        k_train_writer.add_scalar(metric, value, epoch)

                    for metric, value in k_valid_metrics_.items():
                        k_valid_writer.add_scalar(metric, value, epoch)

                if self.earlystopping_ and self.earlystopping_.is_done():
                    break

            k_train_metrics_ = self._evaluate(k_train_dataloader)
            k_valid_metrics_ = self._evaluate(k_valid_dataloader)

            if self.writer_:
                k_train_writer.add_hparams(
                    self.parameters_dict_,
                    {
                        'hparam/' + metric: value
                        for metric, value in k_train_metrics_.items()
                    }
                )
                k_valid_writer.add_hparams(
                    self.parameters_dict_,
                    {
                        'hparam/' + metric: value
                        for metric, value in k_valid_metrics_.items()
                    }
                )

            accumulator.add(
                {
                    'k_train_loss': k_train_metrics_['loss'],
                    'k_valid_loss': k_valid_metrics_['loss'],
                    'k': 1
                }
            )

            if verbose:
                k_train_loss_ = k_train_metrics_['loss']
                k_valid_loss_ = k_valid_metrics_['loss']
                print(
                    f'Fold {k}: train loss {k_train_loss_:.6f}, '
                    f'valid loss {k_valid_loss_:.6f}'
                )

        self.k_train_loss_ = \
            accumulator.get('k_train_loss') / accumulator.get('k')
        self.k_valid_loss_ = \
            accumulator.get('k_valid_loss') / accumulator.get('k')

        if verbose:
            print(
                f'Result: train loss {self.k_train_loss_:.6f}, '
                f'valid loss {self.k_valid_loss_:.6f}'
            )

        return self

    def inference(self, test_dataset, verbose=False):
        r"""Infer the result of test dataset using the trained
        neural network.

        We fetch the results of test dataset using the trained
        neural network stored inside the attribute net. The results
        will be stored inside the attribute of outputs.

        Parameters
        ----------
        test_dataset : torch.utils.data.Dataset
            The test dataset.
        verbose : bool, optional
            If show the progress of inference process.

        Returns
        -------
        outputs : np.ndarray
            Return the outputs.
        """

        if verbose:
            print(
                f'Inferring with parameters: {self.__repr_parameters__()} '
                f'on devices {self.devices_}'
            )

        self.net_.to(self.device_)
        self.loss_fn_.to(self.device_)

        test_dataloader = self.test_loader_fn_(
            test_dataset,
            batch_size=self.batch_size_,
            num_workers=self.num_workers_,
            shuffle=False,
            **self.test_loader_kwargs_
        )

        self.outputs_ = self._inference(test_dataloader, verbose=verbose)

        return self.outputs_

    def latent(self, test_dataset, verbose=False):
        r"""Getting the latent representation of test dataset using
        the trained neural network.

        We fetch the latent representation of test dataset using the
        trained neural network stored inside the attribute net. The
        results will be stored inside the attribute of outputs.

        Parameters
        ----------
        test_dataset : torch.utils.data.Dataset
            The test dataset.
        verbose : bool, optional
            If show the progress of inference process.

        Returns
        -------
        outputs : np.ndarray
            Return the outputs.
        """

        if verbose:
            print(
                'Getting latent representation with '
                f'parameters: {self.__repr_parameters__()}'
            )

        self.net_.to(self.device_)
        self.loss_fn_.to(self.device_)

        test_dataloader = self.test_loader_fn_(
            test_dataset,
            batch_size=self.batch_size_,
            num_workers=self.num_workers_,
            shuffle=False,
            **self.test_loader_kwargs_
        )

        self.outputs_ = self._latent(test_dataloader, verbose=verbose)

        return self.outputs_

    def _save_best_model(self):
        r"""Save the best model.

        Save the best model parameters under the current directory.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.save_model('best_model.pth')

        return self

    def _save_last_model(self):
        r"""Save the last model.

        Save the last model parameters under the current directory.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.save_model('last_model.pth')

        return self

    def save_model(self, model_path):
        r"""Save the model parameters to desired path.

        Parameters
        ----------
        model_path : str
            The path to save model parameters.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if isinstance(self.net_, torch.nn.DataParallel):
            torch.save(
                self.net_.module.state_dict(),
                model_path
            )
        else:
            torch.save(
                self.net_.state_dict(),
                model_path
            )

        return self

    def load_model(self, model_path):
        r"""Load the model parameters from desired path.

        Parameters
        ----------
        model_path : str
            The path to save model parameters.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if isinstance(self.net_, torch.nn.DataParallel):
            self.net_.module.load_state_dict(
                torch.load(model_path, map_location=self.device_)
            )
        else:
            self.net_.load_state_dict(
                torch.load(model_path, map_location=self.device_)
            )

        return self

    def get_summary(self, dataset, verbose=False):
        r"""Get a summary of neural network to be investigated.

        This function will retrieve the inferring speed, as items per
        second to show the neural network inference ability, while
        also show a detailed structure report using torchinfo summary
        report.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset used to get summary report.
        verbose : bool, optional
            If show the progress of inference process.

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.net_.to(self.device_)
        self.loss_fn_.to(self.device_)

        self.net_.eval()

        loader = self.train_loader_fn_(
            dataset,
            batch_size=self.batch_size_,
            num_workers=self.num_workers_,
            shuffle=False,
            **self.train_loader_kwargs_
        )

        sample_batch = next(iter(loader))
        sample_net_input = tuple(
            sample_batch[ind[0]].to(self.device_)
            if len(ind) == 1
            else getattr(sample_batch[ind[0]], ind[1]).to(self.device_)
            for ind in self.net_input_ind_
        )

        start_ts = time.time()
        for batch in tqdm(loader, ascii=True, disable=not verbose):

            if not isinstance(batch, (tuple, list)):
                batch = (batch, )

            net_input = tuple(
                batch[ind[0]].to(self.device_)
                if len(ind) == 1
                else getattr(batch[ind[0]], ind[1]).to(self.device_)
                for ind in self.net_input_ind_
            )

            self.net_(*net_input)
        end_ts = time.time()

        ts = end_ts - start_ts

        self.summary_ = summary(
            self.net_, input_data=sample_net_input,
            verbose=0
        )

        divider = "=" * self.summary_.formatting.get_total_width()

        if verbose:
            print(divider)
            print(
                f'Inferring {len(dataset)} data in {ts:.6f}s '
                f'[{len(dataset) / ts:.6f}it/s]'
            )
            print(
                f'On devices: {self.devices_}'
            )
            print(self.summary_)

        return self

    def __repr_parameters__(self):
        r"""Generate parameters representation string.

        Returns
        -------
        str
            The parameters represented in string format.
        """

        repr_parameters = ', '.join(
            [
                f'{k}={v}'
                for k, v in self.parameters_dict_.items()
            ]
        )

        return repr_parameters

    def __repr__(self):
        r"""Generate representation string for print function.

        Returns
        -------
        str
            The representation in string format.
        """

        return self.__repr_parameters__()
