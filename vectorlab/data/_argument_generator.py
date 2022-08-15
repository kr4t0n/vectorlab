"""
Argument generator is used to generate combinations of
different arguments options. It is often used in grid
search.
"""

from itertools import product, repeat


def kwargs_expansion(kwargs_dict):
    r"""kwargs expansion to generate potential arguments.

    This function helps to generate the full combination of potential
    testing arguments. It receives a dictionary with each testing
    argument and its list of values. This function will expand
    the full combination of these values to generator a official kwargs
    that could be directly fed into algorithms.

    Since the return type is a map, is could be used as

    `[alg(**kwargs) for kwargs in kwargs_expansion(kwargs_dict)]`

    to obtain the results.

    Parameters
    ----------
    kwargs_dict : dict, {key: list of values}
        The dictionary of testing arguments.

    Returns
    -------
    map
        The full combination of arguments values.
    """

    expansion = list(product(*kwargs_dict.values()))
    keys = repeat(list(kwargs_dict.keys()), len(expansion))

    return map(lambda x: dict(zip(x[0], x[1])), zip(keys, expansion))
