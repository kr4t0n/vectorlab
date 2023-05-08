import time
import uuid
import evaluate
import numpy as np

from tqdm.auto import tqdm
from accelerate import Accelerator
from sklearn.model_selection import KFold

import torch

from torchinfo import summary
from torch.utils.data import Subset
from torch_geometric.data import Data, Batch

from ..base import SLMixin, Accumulator
from ._resolver import (
    dataloader_resolver,
    optimizer_resolver, scheduler_resolver, earlystopping_resolver,
    logger_resolver
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
    metrics : list
        The metrics used to evaluate.
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
    logger_fn : str, callable
        The logger to be used to log the metrics.
    logger_project : str
        The name of the current project.
    logger_id : str
        The name of the current experiment id.
    logger_freq : int
        The frequency to log the metrics.
    device : str, list, optional
        If given specified device or a list of devices, the explorer will use
        such device(s) as desired. If no device is specified, the explorer
        will automatically choose the device(s).
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
    metrics_ : list
        The metrics used to evaluate.
    epochs_len_ : int
        The length of epoch number.
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
    scheduler_fn_ : torch.optim.lr_scheduler
        The given callable function to initialize corresponding scheduler.
    scheduler_kwargs_ : dict
        The extra arguments for initialize the scheduler.
    scheduler_ : torch.optim.lr_scheduler
        The scheduler to decay the learning rate of optimizer.
    earlystopping_fn_ : callabe
        The earlystopping to be used to avoid overfitting.
    earlystopping_metric_ : str
        The montior metric for earlystopping.
    earlystopping_kwarg_ : dict
        The extra arguments for initialize the earlystopping.
    earlystopping_ : callable
        The earlystopping to break epoch iteration.
    logger_fn_ : str, callable
        The logger to be used to log the metrics.
    logger_project_ : str
        The name of the current project.
    logger_id_ : str
        The name of the current experiment id.
    logger_freq_ : int
        The frequency to log the metrics.
    device_ : torch.device
        The first device used to store the neural network and data.
    devices_ : list
        The list of potential devices that would be used.
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

    def __init__(self,
                 net, loss_fn,
                 batch_input, net_input, loss_input,
                 k=5,
                 batch_size=32, num_workers=0, num_epochs=100,
                 extra_metrics=None,
                 train_loader_fn='dataloader', train_loader_kwargs=None,
                 valid_loader_fn=None, valid_loader_kwargs=None,
                 test_loader_fn=None, test_loader_kwargs=None,
                 optimizer_fn='adamw',
                 learning_rate=0.1, weight_decay=0, optimizer_kwargs=None,
                 scheduler_fn='cosine_annealing_lr',
                 scheduler_kwargs=None,
                 earlystopping_fn='desc_es',
                 earlystopping_metric='loss', earlystopping_kwargs=None,
                 logger_fn='tensorboard',
                 logger_project='vectorlab_explorer_project', logger_id=None,
                 logger_freq=1,
                 parameters_dict=None):

        super().__init__()

        self.k_ = int(k)
        self.batch_size_ = int(batch_size)
        self.num_workers_ = int(num_workers)
        self.num_epochs_ = int(num_epochs)
        self.epochs_len_ = int(np.floor(np.log10(self.num_epochs_) + 1))

        if extra_metrics is None:
            self.extra_metrics_ = None
        else:
            self.extra_metrics_ = evaluate.combine(extra_metrics)

        self.train_loader_fn_ = dataloader_resolver(train_loader_fn)
        self.train_loader_kwargs_ = \
            train_loader_kwargs if train_loader_kwargs else {}

        if valid_loader_fn is None:
            self.valid_loader_fn_ = self.train_loader_fn_
        else:
            self.valid_loader_fn_ = dataloader_resolver(valid_loader_fn)
        self.valid_loader_kwargs_ = \
            valid_loader_kwargs if valid_loader_kwargs else {}

        if test_loader_fn is None:
            self.test_loader_fn_ = self.train_loader_fn_
        else:
            self.test_loader_fn_ = dataloader_resolver(test_loader_fn)
        self.test_loader_kwargs_ = \
            test_loader_kwargs if test_loader_kwargs else {}

        self.net_ = net
        self.loss_fn_ = loss_fn

        self.batch_input_ = batch_input
        self.net_input_ = net_input
        self.loss_input_ = loss_input
        self._init_input_format()

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

        self.logger_fn_ = logger_fn
        self.logger_project_ = logger_project
        self.logger_id_ = logger_id
        self.logger_freq_ = logger_freq
        self._init_logger()

        self._init_parameters_dict_(parameters_dict)

        self.accelerator_ = Accelerator()
        self.net_, self.loss_fn_ = self.accelerator_.prepare(
            self.net_, self.loss_fn_
        )
        self.optimizer_, self.scheduler_ = self.accelerator_.prepare(
            self.optimizer_, self.scheduler_
        )

        return

    def _init_optimizer(self):
        r"""Initialize proper optimizer

        Returns
        -------
        self : Explorer
            Return itself.
        """

        self.optimizer_fn_ = optimizer_resolver(self.optimizer_fn_)

        self.optimizer_kwargs_ = \
            self.optimizer_kwargs_ if self.optimizer_kwargs_ else {}

        # update learning rate and weight decay
        self.optimizer_kwargs_.update(
            {
                'lr': self.learning_rate_,
                'weight_decay': self.weight_decay_
            }
        )

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

        self.scheduler_fn_ = scheduler_resolver(self.scheduler_fn_)

        if self.scheduler_kwargs_ is None:
            if self.scheduler_fn_ == \
                    torch.optim.lr_scheduler.StepLR:
                self.scheduler_kwargs_ = {
                    'step_size': self.num_epochs_ // 4
                }
            elif self.scheduler_fn_ == \
                    torch.optim.lr_scheduler.CosineAnnealingLR:
                self.scheduler_kwargs_ = {
                    'T_max': self.num_epochs_
                }
            else:
                self.scheduler_kwargs_ = {}
        else:
            self.scheduler_kwargs_ = self.scheduler_kwargs_

        if self.scheduler_fn_ is None:
            self.scheduler_ = None
        else:
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

        self.earlystopping_fn_ = earlystopping_resolver(self.earlystopping_fn_)

        self.earlystopping_kwargs_ = \
            self.earlystopping_kwargs_ if self.earlystopping_kwargs_ else {}

        if self.earlystopping_fn_ is None:
            self.earlystopping_ = None
        else:
            self.earlystopping_ = self.earlystopping_fn_(
                **self.earlystopping_kwargs_
            )

        return self

    def _init_logger(self):
        r"""Initialize proper logger

        Returns
        -------
        self : Explorer
            Return itself.
        """
        self.logger_fn_ = logger_resolver(self.logger_fn_)

        if self.logger_id_ is None:
            self.logger_id_ = str(uuid.uuid4())

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

    def _generate_input(self, batch):
        r"""Generate net input and loss input from giver batch.

        Parameters
        ----------
        batch : tuple
            Tuple of tensors to be a batch.

        Returns
        -------
        net_input : tuple
            Tuple of tensors to be a net input.
        loss_input : tuple
            Tuple of tensors to be a loss input.
        """

        net_input = tuple(
            batch[ind[0]]
            if len(ind) == 1
            else getattr(batch[ind[0]], ind[1])
            for ind in self.net_input_ind_
        )
        loss_input = tuple(
            batch[ind[0]]
            if len(ind) == 1
            else getattr(batch[ind[0]], ind[1])
            for ind in self.loss_input_ind_
        )

        return net_input, loss_input

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

    def _pbar_disable(self, verbose):
        r"""If pbar display should be disabled.

        Parameters
        ----------
        verbose : int
            Level of verbose mode.

        Returns
        -------
        bool
            Return if pbar display should be disabled.
        """

        if (verbose > 1) and self.accelerator_.is_local_main_process:
            return False
        else:
            return True

    def _calc_metrics(self, verbose):
        r"""If calculate metrics after each epoch.

        Parameters
        ----------
        verbose : int
            Level of verbose mode.

        Returns
        -------
        bool
            Return if metrics should be calculated.
        """

        if (verbose >= 1) or self.earlystopping_fn_ or self.logger_fn_:
            return True
        else:
            return False

    def reset(self):
        r"""The reset method.

        This method will reset the parameters inside the neural network
        and re-initialized the optimizer and scheduler.

        Returns
        -------
        self : Explorer
            Return itself.
        """

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

        self.optimizer_, self.scheduler_ = self.accelerator_.prepare(
            self.optimizer_, self.scheduler_
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

            net_input, loss_input = self._generate_input(batch)

            self.optimizer_.zero_grad()

            output = self.net_(*net_input)
            output = output if isinstance(output, tuple) else (output, )

            loss = self.loss_fn_(*output, *loss_input)
            self.accelerator_.backward(loss)
            self.optimizer_.step()

        if self.scheduler_:
            self.scheduler_.step()
        if self.earlystopping_:
            self.earlystopping_.step()

        return self

    def _infer_n_samples(self, batch, loader):
        r"""Infer number of samples inside the batch.

        Parameters
        ----------
        batch : tensor, Data, Batch
            The batch of data.
        loader : torch.utils.data.DataLoader
            The data loader generate corresponding batch.

        Returns
        -------
        n_samples : int
            The number of samples inside.
        """

        n_samples = 0

        # NN
        if isinstance(batch, torch.Tensor):
            # check if loader has attribute batch_first
            # in case it is a sequence data
            if hasattr(loader, 'batch_first'):
                if loader.batch_first:
                    n_samples = batch.shape[0]
                else:
                    n_samples = batch.shape[1]
            else:
                # normal dataloader, first dimension is usually
                # the number of samples
                n_samples = batch.shape[0]

        # GNN
        if isinstance(batch, Data):
            # pytorch_geometric Data type for node level tasks
            n_samples = batch.num_nodes
        elif isinstance(batch, Batch):
            # pytorch_geometric Batch type for graph level tasks
            n_samples = batch.num_graphs

        return n_samples

    def _extra_metrics_transform(self, output, loss_input):
        r"""Some extra transformations performed to calculate metrics.

        Output is in the tuple format of the neural network output, while
        loss_input is in the tuple format to feed in to loss function.

        In most cases, the first element of output should be the predictions,
        and the first element of loss_input should be the references. However,
        if you have more flexible format, you can overload this function as
        you desired.

        Parameters
        ----------
        output : tuple
            The output of the neural network.
        loss_input : tuple
            The input of the loss function.

        Returns
        -------
        predictions : tensor
            The predictions of the neural network.
        references : tensor
            The references of the loss function.
        """

        predictions = output[0].detach()
        references = loss_input[0].detach()

        return predictions, references

    def _evaluate(self, loader):
        r"""This evaluation method.

        We calculate the overall loss as our evaluation metrics, while the
        way of calculating the loss is same as the process in the train part.
        Besides, some extra metrics are also calculated during this process.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.

        Returns
        -------
        metrics : dict
            The dictionary contained the evaluation metrics.
        """

        self.net_.eval()

        accumulator = Accumulator(2, ['loss', 'n_samples'])

        with torch.no_grad():

            for batch in loader:

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                n_samples = self._infer_n_samples(batch[0], loader)

                net_input, loss_input = self._generate_input(batch)

                output = self.net_(*net_input)
                output = output if isinstance(output, tuple) else (output, )

                output, loss_input = self.accelerator_.gather_for_metrics(
                    (output, loss_input)
                )
                loss = self.loss_fn_(*output, *loss_input).item()

                accumulator.add(
                    {
                        'loss': loss * n_samples,
                        'n_samples': n_samples
                    }
                )

                if self.extra_metrics_ is not None:
                    predictions, references = self._extra_metrics_transform(
                        output, loss_input
                    )
                    self.extra_metrics_.add_batch(
                        predictions=predictions,
                        references=references
                    )

        metrics = {
            'loss': accumulator.get('loss') / accumulator.get('n_samples')
        }

        if self.extra_metrics_ is not None:
            metrics.update(self.extra_metrics_.compute())

        return metrics

    def _metrics(self, train_loader, valid_loader=None):
        """Return the evaluated metrics from train and valid loader.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The data loader containing training data to evaluate the trained
            neural network
        valid_loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.

        Returns
        -------
        metrics : dict
            The dictionary contained the evaluation metrics for verfication.
            If valid_loader is provided, it will be valid_metrics, otherwise,
            it will be train_metrics.
        nested_metrics : dict
            The dictionary contained the evaluation metrics. The metrics are
            in a nested representation of
            {'train': {train_metrics}, 'valid': {valid_metrics}}.
        """

        train_metrics = self._evaluate(train_loader)

        if valid_loader is None:
            metrics = train_metrics
            nested_metrics = {'train': train_metrics}

            return metrics, nested_metrics

        valid_metrics = self._evaluate(valid_loader)
        metrics = valid_metrics
        nested_metrics = {'train': train_metrics, 'valid': valid_metrics}

        return metrics, nested_metrics

    def _inference(self, loader, verbose=0):
        r"""This is the abstract inference method.

        This is the abstract inference method that should be implemented in
        children class, that it will inference the result of inputs using
        trained neural network.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.
        verbose : int, optional
            If show the progress of inference process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of inferring all data

        Returns
        -------
        outputs : np.ndarray
            The outputs inferred by trained neural network.
        """

        outputs = []

        self.net_.eval()

        with torch.no_grad():

            for batch in tqdm(loader,
                              ascii=True,
                              disable=self._pbar_disable(verbose)):

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                net_input, _ = self._generate_input(batch)

                output = self.net_(*net_input)
                output = self.accelerator_.gather_for_metrics(output)
                outputs.append(output.detach().cpu().tolist())

        outputs = np.concatenate(outputs)

        return outputs

    def _latent(self, loader, verbose=0):
        r"""This is the abstract latent method.

        This is the abstract getting latent method that should be implemented
        in children class, that it will get latent representation from inputs
        using trained neural network sub-module, usually an encoder.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader containing validation data to evaluate the trained
            neural network.
        verbose : int, optional
            If show the progress of inference process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of inferring all data

        Returns
        -------
        outputs : np.ndarray
            The outputs inferred by trained neural network.
        """

        outputs = []

        self.net_.eval()

        with torch.no_grad():

            for batch in tqdm(loader,
                              ascii=True,
                              disable=self._pbar_disable(verbose)):

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                net_input, _ = self._generate_input(batch)

                output = self.net_.forward_latent(*net_input)
                output = self.accelerator_.gather_for_metrics(output)
                outputs.append(output.detach().cpu().tolist())

        outputs = np.concatenate(outputs)

        return outputs

    def train(
        self,
        train_dataset, valid_dataset=None,
        save_best=False, save_last=True,
        verbose=0
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
        verbose : int, optional
            If show the progress of training process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of training all data

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if verbose and self.accelerator_.is_local_main_process:
            print(
                f'Training with parameters: {self.__repr_parameters__()} '
                f'on devices {self.accelerator_.device}'
            )

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
        else:
            valid_loader = None

        train_loader, valid_loader = self.accelerator_.prepare(
            train_loader, valid_loader
        )

        self.best_loss_ = np.Inf

        if self.accelerator_.is_main_process:

            if self.logger_fn_ is not None:
                logger = self.logger_fn_(
                    self.logger_project_, self.logger_id_,
                    freq=self.logger_freq_
                )
                logger.watch(self.net_, self.loss_fn_)
            else:
                logger = None

        pbar = tqdm(
            range(self.num_epochs_),
            ascii=True,
            disable=self._pbar_disable(verbose)
        )
        for epoch in pbar:
            self._train(train_loader)

            if self._calc_metrics(verbose) or save_best:
                metrics_, nested_metrics_ = self._metrics(
                    train_loader, valid_loader
                )
                metric_repr_ = ', '.join(
                    [
                        f'{key}: {value:.3f}'
                        for key, value in metrics_.items()
                    ]
                )

                pbar.set_description(
                    f'Epoch: {epoch:0{self.epochs_len_}d}, {metric_repr_}'
                )

            if self.accelerator_.is_main_process:

                if self.earlystopping_:
                    self.earlystopping_.record_metric(
                        metrics_[self.earlystopping_metric_]
                    )

                if logger and (epoch % logger.freq_ == 0):
                    logger.log(nested_metrics_, step=epoch)

            if save_best:
                if metrics_['loss'] < self.best_loss_:
                    self.best_loss_ = metrics_['loss']
                    self._save_best_model()

            if self.earlystopping_ and self.earlystopping_.is_done():
                self.accelerator_.wait_for_everyone()
                break

        _, nested_metrics_ = self._metrics(
            train_loader, valid_loader
        )

        self.train_loss_ = nested_metrics_['train']['loss']

        if valid_loader:
            self.valid_loss_ = nested_metrics_['valid']['loss']
        else:
            self.valid_loss_ = np.Inf

        if self.accelerator_.is_main_process:

            if logger:
                logger.log_params(self.parameters_dict_, nested_metrics_)
                logger.unwatch(self.net_)
                logger.close()

        if save_last:
            self._save_last_model()

        if verbose:
            print(
                f'Result: train loss {self.train_loss_:.6f}, '
                f'valid loss {self.valid_loss_:.6f}'
            )

        return self

    def k_fold_train(self, train_dataset, verbose=0):
        r"""Trained the neural network using k fold cross validation.

        Train the neural network using the pre-defined arguments with
        k fold cross validation manner. The average of training loss will
        be stored inside the attribute k_train_loss, while the average of
        validation loss will be stored inside the attribute k_valid_loss.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            The training dataset.
        verbose : int, optional
            If show the progress of k-fold training process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of training all data

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if verbose and self.accelerator_.is_local_main_process:
            print(
                f'Training with parameters: k={self.k_}, '
                f'{self.__repr_parameters__()} '
                f'on devices {self.accelerator_.device}'
            )

        accumulator = Accumulator(3, ['k_train_loss', 'k_valid_loss', 'k'])

        kf = KFold(n_splits=self.k_)
        for k, index in enumerate(kf.split(train_dataset)):

            self.reset()

            (train_index, valid_index) = index

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

            k_train_dataloader, k_valid_dataloader = self.accelerator_.prepare(
                k_train_dataloader, k_valid_dataloader
            )

            if self.accelerator_.is_main_process:

                if self.logger_fn_ is not None:
                    logger = self.logger_fn_(
                        self.logger_project_, f'{self.logger_id_}-kfold-{k}',
                        freq=self.logger_freq_
                    )
                    logger.watch(self.net_, self.loss_fn_)
                else:
                    logger = None

            pbar = tqdm(
                range(self.num_epochs_),
                ascii=True,
                disable=self._pbar_disable(verbose)
            )
            for epoch in pbar:
                self._train(k_train_dataloader)

                if self._calc_metrics(verbose):
                    k_metrics_, k_nested_metrics_ = self._metrics(
                        k_train_dataloader, k_valid_dataloader
                    )
                    metric_repr_ = ', '.join(
                        [
                            f'{key}: {value:.3f}'
                            for key, value in k_metrics_.items()
                        ]
                    )

                    pbar.set_description(
                        f'Epoch: {epoch:0{self.epochs_len_}d}, {metric_repr_}'
                    )

                if self.accelerator_.is_main_process:

                    if self.earlystopping_:
                        self.earlystopping_.record_metric(
                            k_metrics_[self.earlystopping_metric_]
                        )

                    if logger and (epoch % logger.freq_ == 0):
                        logger.log(k_nested_metrics_, step=epoch)

                if self.earlystopping_ and self.earlystopping_.is_done():
                    self.accelerator_.wait_for_everyone()
                    break

            _, k_nested_metrics_ = self._metrics(
                k_train_dataloader, k_valid_dataloader
            )

            if self.accelerator_.is_main_process:

                if logger:
                    logger.log_params(self.parameters_dict_, k_nested_metrics_)
                    logger.unwatch(self.net_)
                    logger.close()

            accumulator.add(
                {
                    'k_train_loss': k_nested_metrics_['train']['loss'],
                    'k_valid_loss': k_nested_metrics_['valid']['loss'],
                    'k': 1
                }
            )

            if verbose:
                k_train_loss_ = k_nested_metrics_['train']['loss']
                k_valid_loss_ = k_nested_metrics_['valid']['loss']
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

    def inference(self, test_dataset, verbose=0):
        r"""Infer the result of test dataset using the trained
        neural network.

        We fetch the results of test dataset using the trained
        neural network stored inside the attribute net. The results
        will be stored inside the attribute of outputs.

        Parameters
        ----------
        test_dataset : torch.utils.data.Dataset
            The test dataset.
        verbose : int, optional
            If show the progress of inference process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of inferring all data

        Returns
        -------
        outputs : np.ndarray
            Return the outputs.
        """

        if verbose and self.accelerator_.is_local_main_process:
            print(
                f'Inferring with parameters: {self.__repr_parameters__()} '
                f'on devices {self.accelerator_.device}'
            )

        test_dataloader = self.test_loader_fn_(
            test_dataset,
            batch_size=self.batch_size_,
            num_workers=self.num_workers_,
            shuffle=False,
            **self.test_loader_kwargs_
        )
        test_dataloader = self.accelerator_.prepare(test_dataloader)

        self.outputs_ = self._inference(test_dataloader, verbose=verbose)

        return self.outputs_

    def latent(self, test_dataset, verbose=0):
        r"""Getting the latent representation of test dataset using
        the trained neural network.

        We fetch the latent representation of test dataset using the
        trained neural network stored inside the attribute net. The
        results will be stored inside the attribute of outputs.

        Parameters
        ----------
        test_dataset : torch.utils.data.Dataset
            The test dataset.
        verbose : int, optional
            If show the progress of inference process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of inferring all data

        Returns
        -------
        outputs : np.ndarray
            Return the outputs.
        """

        if verbose and self.accelerator_.is_local_main_process:
            print(
                'Getting latent representation with '
                f'parameters: {self.__repr_parameters__()}'
            )

        test_dataloader = self.test_loader_fn_(
            test_dataset,
            batch_size=self.batch_size_,
            num_workers=self.num_workers_,
            shuffle=False,
            **self.test_loader_kwargs_
        )
        test_dataloader = self.accelerator_.prepare(test_dataloader)

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

        self.accelerator_.wait_for_everyone()

        if self.accelerator_.is_main_process:
            unwrapped_net = self.accelerator_.unwrap_model(self.net_)
            self.accelerator_.save(unwrapped_net.state_dict(), model_path)

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

        if self.accelerator_.is_main_process:
            unwrapped_net = self.accelerator_.unwrap_model(self.net_)
            unwrapped_net.load_state_dict(torch.load(model_path))

        return self

    def get_summary(self, dataset=None, verbose=1):
        r"""Get a summary of neural network to be investigated.

        This function will retrieve the inferring speed, as items per
        second to show the neural network inference ability, while
        also show a detailed structure report using torchinfo summary
        report.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset used to get summary report.
        verbose : int, optional
            If show the progress of get summary process.
            0 : quite mode
            1 : show basic summary
            2 : show progress bar of inferring all data

        Returns
        -------
        self : Explorer
            Return itself.
        """

        if dataset is None:
            if self.accelerator_.is_local_main_process:
                print(
                    'Dataset is not provided, you can only get basic '
                    'structure of neural network. For more detailed '
                    'information, you should provide a sample dataset.'
                )

        self.net_.eval()

        if dataset is None:
            loader = None
            sample_net_input = None
        else:
            loader = self.train_loader_fn_(
                dataset,
                batch_size=self.batch_size_,
                num_workers=self.num_workers_,
                shuffle=False,
                **self.train_loader_kwargs_
            )
            loader = self.accelerator_.prepare(loader)

            sample_batch = next(iter(loader))

            if not isinstance(sample_batch, (tuple, list)):
                sample_batch = (sample_batch, )

            sample_net_input, _ = self._generate_input(sample_batch)

            start_ts = time.time()
            for batch in tqdm(loader,
                              ascii=True,
                              disable=self._pbar_disable(verbose)):

                if not isinstance(batch, (tuple, list)):
                    batch = (batch, )

                net_input, _ = self._generate_input(batch)

                self.net_(*net_input)
            end_ts = time.time()

            ts = end_ts - start_ts

        if self.accelerator_.is_local_main_process:

            self.summary_ = summary(
                self.net_,
                input_data=sample_net_input,
                verbose=0
            )

            divider = "=" * self.summary_.formatting.get_total_width()

            if verbose:
                if dataset is None:
                    print(self.summary_)
                else:
                    print(divider)
                    print(
                        f'Inferring {len(dataset)} data in {ts:.6f}s '
                        f'[{len(dataset) / ts:.6f}it/s] \n'
                        f'On devices: {self.accelerator_.device}'
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
