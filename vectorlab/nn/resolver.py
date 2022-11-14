import torch

from . import _earlystopping


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
    object
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
    object
        The resolved scheduler class.
    """

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
    object
        The resolved earlystopping class.
    """

    base_class = _earlystopping._EarlyStopping  # noqa

    earlystoppings = [
        es
        for es in vars(_earlystopping).values()
        if isinstance(es, type) and issubclass(es, base_class)
    ]
    earlystoppings_dict = {}

    return resolver(query, earlystoppings, earlystoppings_dict)
