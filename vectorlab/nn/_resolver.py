import torch
import torch_geometric
import importlib.metadata as importlib_metadata

from packaging.version import parse

from . import _earlystopping, _logger
from ..data import dataloader as _dataloader

_torch_version = importlib_metadata.version('torch')


def normalize_string(s):
    r"""Normalize an input string.

    This function will normalize a string, remove all '-', '_', ' '.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Normalized string.
    """

    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(query, classes, classes_dict):
    r"""A class resolver.

    Parameters
    ----------
    query : str
        Input class query.
    classes : list
        A list of potential classes.
    classes_dict : dict
        A dictionary of potential classes.

    Returns
    -------
    object
        The resolved class object.

    Raises
    ------
    ValueError
        If query cannot be resolved by potential classes,
        a ValueError is raised.
    """

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)

        if query_repr == cls_repr:
            obj = cls
            return obj

    for cls_repr, cls in classes_dict.items():
        cls_repr = normalize_string(cls_repr)

        if query_repr == cls_repr:
            obj = cls
            return obj

    classes_repr = \
        set(classes_dict.keys()) | set(cls.__name__ for cls in classes)

    raise ValueError(
        f'Variable resolver option does not match the constraint. '
        f'Potential options are {classes_repr}, your option is {query}.'
    )


def dataloader_resolver(query):
    r"""A dataloader resolver

    List all dataloaders supported by vectorlab.data.dataloader and
    find corresponding dataloader.

    Parameters
    ----------
    query : str
        Input dataloader query.

    Returns
    -------
    torch.utils.data.DataLoader
        The resolved dataloader class.
    """

    base_class = torch.utils.data.DataLoader

    dataloaders = [
        dataloader
        for dataloader in vars(_dataloader).values()
        if isinstance(dataloader, type) and issubclass(dataloader, base_class)
    ]
    dataloaders_dict = {}

    return resolver(query, dataloaders, dataloaders_dict)


def optimizer_resolver(query):
    """An optimizer resolver.

    List all optimizers suppored by torch.optim and find
    corresponding optimizer.

    Parameters
    ----------
    query : str
        Input optimizer query.

    Returns
    -------
    torch.optim.Optimizer
        The resolved optimizer class.
    """

    base_class = torch.optim.Optimizer

    optimizers = [
        optimizer
        for optimizer in vars(torch.optim).values()
        if isinstance(optimizer, type) and issubclass(optimizer, base_class)
    ]
    optimizers_dict = {}

    return resolver(query, optimizers, optimizers_dict)


def scheduler_resolver(query):
    r"""A scheduler resolver.

    List all schedulers supported by torch.optim.lr_scheduler and find
    corresponding scheduler.

    Parameters
    ----------
    query : str
        Input scheduler query.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        The resolved scheduler class.
    """

    if parse(_torch_version) >= parse('2.0.0'):
        base_class = torch.optim.lr_scheduler.LRScheduler
    else:
        base_class = torch.optim.lr_scheduler._LRScheduler  # noqa

    schedulers = [
        scheduler
        for scheduler in vars(torch.optim.lr_scheduler).values()
        if isinstance(scheduler, type) and issubclass(scheduler, base_class)
    ]
    schedulers_dict = {}

    return resolver(query, schedulers, schedulers_dict)


def earlystopping_resolver(query):
    r"""A earlystopping resolver.

    List all earlystoppings supported by vectorlab.nn._earlystopping and find
    corresponding earlystopping.

    Parameters
    ----------
    query : str
        Input earlystopping query.

    Returns
    -------
    _earlystopping.EarlyStopping
        The resolved earlystopping class.
    """

    base_class = _earlystopping.EarlyStopping

    earlystoppings = [
        es
        for es in vars(_earlystopping).values()
        if isinstance(es, type) and issubclass(es, base_class)
    ]
    earlystoppings_dict = {}

    return resolver(query, earlystoppings, earlystoppings_dict)


def logger_resolver(query):
    r"""A logger resolver.

    List all loggers supported by vectorlab.nn._logger and find
    corresponding logger.

    Parameters
    ----------
    query : str
        Input logger query.

    Returns
    -------
    _logger.BaseLogger
        The resolved logger class.
    """

    base_class = _logger.BaseLogger

    loggers = [
        logger
        for logger in vars(_logger).values()
        if isinstance(logger, type) and issubclass(logger, base_class)
    ]
    loggers_dict = {}

    return resolver(query, loggers, loggers_dict)


def activation_resolver(query):
    r"""An activation resolver.

    List all activations supported by torch.nn.modules.activation and
    find corresponding activation.

    Parameters
    ----------
    query : str
        Input activation query.

    Returns
    -------
    torch.nn.Module
        The resolved activation class.
    """

    base_class = torch.nn.Module

    activations = [
        activation
        for activation in vars(torch.nn.modules.activation).values()
        if isinstance(activation, type) and issubclass(activation, base_class)
    ]
    activations_dict = {}

    return resolver(query, activations, activations_dict)


def nn_normalization_resolver(query):
    r"""A nn batch normalization resolver.

    List all batch normalizations supported by torch.nn.modules.batchnorm and
    find corrresponding batch normalization.

    Parameters
    ----------
    query : str
        Input nn batch normalization query.

    Returns
    -------
    torch.nn.Module
        The resolved nn batch normalization class.
    """

    base_class = torch.nn.Module

    normalizations = [
        norm
        for norm in vars(torch.nn.modules.batchnorm).values()
        if isinstance(norm, type) and issubclass(norm, base_class)
    ]
    normalizations_dict = {}

    return resolver(query, normalizations, normalizations_dict)


def gnn_normalization_resolver(query):
    r"""A gnn batch normalization resolver.

    List all batch normalization supported by torch_geometric.nn.norm and
    find corresponding batch normalization.

    Parameters
    ----------
    query : str
        Input gnn batch normalization query.

    Returns
    -------
    torch.nn.Module
        The resolved gnn batch normalization class.
    """

    base_class = torch.nn.Module

    normalizations = [
        norm
        for norm in vars(torch_geometric.nn.norm).values()
        if isinstance(norm, type) and issubclass(norm, base_class)
    ]
    normalizations_dict = {}

    return resolver(query, normalizations, normalizations_dict)
