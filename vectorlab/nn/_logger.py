import uuid
import wandb

from torch.utils.tensorboard import SummaryWriter

from ..base import SLMixin


class BaseLogger(SLMixin):
    r"""The BaseLogger class to log training metrics.

    The logger is aimed to monitor and log the hyper-paramters
    and metrics during training.

    Parameters
    ----------
    project : str
        The name of the current project.
    exp_id : str
        The name of the current experiment id.
    freq : int
        The frequency to log the training metrics.
    """

    def __init__(self, project, exp_id=None, freq=1):

        super().__init__()

        self.project_ = project
        self.exp_id_ = exp_id if exp_id else str(uuid.uuid4())
        self.freq_ = freq

        return

    def _flatten_metrics(self, metrics):
        r"""Flatten the nested metrics.

        The metrics should be in nested format, in a form of
        {'train': {train_metrics}, 'valid': {valid_metrics}, ...}.
        This flatten function will convert the nested metrics into
        {'train/...': '...', 'valid/...': ...}.

        Parameters
        ----------
        metrics : dict
            The logged training metrics.

        Returns
        -------
        flattened_metrics : dict
            The flattened training metrics.
        """

        # metrics is in format of
        # {'train': {train_metrics}, 'valid': {valid_metrics}, ...}

        flattened_metrics = {
            f'{tag}/{metric_key}': metric_value
            for tag, tag_metrics in metrics.items()
            for metric_key, metric_value in tag_metrics.items()
        }

        return flattened_metrics

    def watch(self, net, loss_fn):
        r"""Watch the model gradients.

        This could be useful for wandb logger to monitor the model
        gradients.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network used in training.
        loss_fn : torch.nn.Module
            The loss function used in training.
        """
        raise NotImplementedError

    def unwatch(self, net):
        r"""Un-watch the model gradients.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network used in training.
        """
        raise NotImplementedError

    def log(self, metrics, step):
        r"""Log the training metrics.

        metrics is in from of
        {'train': {train_metrics}, 'valid': {valid_metrics}, ...}

        Parameters
        ----------
        metrics : dict
            The current training metrics.
        step : int
            The current step during the training.
        """
        raise NotImplementedError

    def log_params(self, params, metrics=None):
        r"""Log the hyper-parameters and last possible training metrics.

        metrics is in from of
        {'train': {train_metrics}, 'valid': {valid_metrics}, ...}

        Parameters
        ----------
        params : dict
            The hyper-parameters used for training.
        metrics : dict, optional
            The last possible training metrics.
        """
        raise NotImplementedError

    def close(self):
        r"""Close the logger.
        """
        raise NotImplementedError


class Tensorboard(BaseLogger):
    r"""The Tensorboard class to log training metrics.

    Parameters
    ----------
    project : str
        The name of the current project.
    exp_id : str
        The name of the current experiment id.
    freq : int
        The frequency to log the training metrics.
    """

    def __init__(self, project, exp_id=None, freq=1):

        super().__init__(project, exp_id, freq)

        self.logger_ = SummaryWriter(
            log_dir=f'{self.project_}/{self.exp_id_}'
        )

        return

    def watch(self, net, loss_fn):
        r"""Watch the model gradients.

        Currently not supported in tensorboard.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network used in training.
        loss_fn : torch.nn.Module
            The loss function used in training.
        """
        return

    def unwatch(self, net):
        r"""Un-watch the model gradients.

        Currently not supported in tensorboard.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network used in training.
        """
        return

    def log(self, metrics, step):
        r"""Log the training metrics.

        metrics is in from of
        {'train': {train_metrics}, 'valid': {valid_metrics}, ...}

        Parameters
        ----------
        metrics : dict
            The current training metrics.
        step : int
            The current step during the training.
        """

        for tag, tag_metrics in metrics.items():
            self.logger_.add_scalars(
                tag, tag_metrics,
                global_step=step
            )

        return

    def log_params(self, params, metrics=None):
        r"""Log the hyper-parameters and last possible training metrics.

        metrics is in from of
        {'train': {train_metrics}, 'valid': {valid_metrics}, ...}

        Parameters
        ----------
        params : dict
            The hyper-parameters used for training.
        metrics : dict, optional
            The last possible training metrics.
        """

        self.logger_.add_hparams(
            params,
            self._flatten_metrics(metrics)
        )

        return

    def close(self):
        r"""Close the logger.
        """

        self.logger_.close()

        return


class Wandb(BaseLogger):
    r"""The Wandb class to log training metrics.

    Parameters
    ----------
    project : str
        The name of the current project.
    exp_id : str
        The name of the current experiment id.
    freq : int
        The frequency to log the training metrics.
    """

    def __init__(self, project, exp_id=None, freq=1):

        super().__init__(project, exp_id, freq)

        wandb.init(
            project=self.project_,
            id=self.exp_id_
        )

        return

    def watch(self, net, loss_fn):
        r"""Watch the model gradients.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network used in training.
        loss_fn : torch.nn.Module
            The loss function used in training.
        """

        wandb.watch(
            net, loss_fn,
            log='all',
            log_freq=self.freq_
        )

        return

    def unwatch(self, net):
        r"""Un-watch the model gradients.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network used in training.
        """

        wandb.unwatch(net)

        return

    def log(self, metrics, step):
        r"""Log the training metrics.

        metrics is in from of
        {'train': {train_metrics}, 'valid': {valid_metrics}, ...}

        Parameters
        ----------
        metrics : dict
            The current training metrics.
        step : int
            The current step during the training.
        """

        wandb.log(
            self._flatten_metrics(metrics),
            step=step
        )

        return

    def log_params(self, params, metrics=None):
        r"""Log the hyper-parameters and last possible training metrics.

        Wandb can automatrically log the last possible trianing metrics.

        Parameters
        ----------
        params : dict
            The hyper-parameters used for training.
        metrics : dict, optional
            The last possible training metrics.
        """

        config = wandb.config

        for param, value in params.items():
            setattr(config, param, value)

        return

    def close(self):
        r"""Close the logger.
        """

        wandb.finish()

        return
