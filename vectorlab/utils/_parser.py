"""
Parse various files.
"""

import os
import yaml
import dill

from ._check import check_valid_option


def convert_conditions(conditions, serialized=False):
    r"""Minvera convert condition will convert the conditions into
    condition lambda function.

    We want to convert the man written condition inside the yaml file
    to convert to an actual lambda function which could be used on
    real data.

    Inside conditions, every condition must include a type and a value.
    The type should be one of the following types:

    - lt
    - lte
    - gt
    - gte

    Parameters
    ----------
    conditions : dict
        The dictionary containing key and corresponding with conditions.
    serialized : bool, optional
        If serialized the lambda function so it can be used to share the
        conditions across processes when using serialized methods.

    Returns
    -------
    conditions : dict
        New conditions with lambda function inside.
    """

    for name, condition in conditions.items():

        condition_type = condition['type']
        condition_value = condition['value']

        condition_type = check_valid_option(
            condition_type,
            options=['lt', 'lte', 'gt', 'gte'],
            variable_name=f'condition type of {name}'
        )

        if condition_type == 'lt':
            conditions[name] = lambda x: x < condition_value
        elif condition_type == 'lte':
            conditions[name] = lambda x: x <= condition_value
        elif condition_type == 'gt':
            conditions[name] = lambda x: x > condition_value
        elif condition_type == 'gte':
            conditions[name] = lambda x: x >= condition_value

        if serialized:
            conditions[name] = dill.dumps(conditions[name])

    return conditions


def parse_yaml_config(yaml_file):
    r"""Minerva parse YAML config function will parse YAML file to
    generate a configuration dictionary.

    Parameters
    ----------
    yaml_file : str
        The YAML file path to be parsed.

    Returns
    -------
    config : dictionary
        The configuration dictionary parsed from YAML file.

    Raises
    ------
    ValueError
        When yaml_file does not exist or when it is not a file,
        a ValueError is raised.
    """

    if not os.path.exists(yaml_file):
        raise ValueError(
            f'{yaml_file} YAML file path error'
        )

    if not os.path.isfile(yaml_file):
        raise ValueError(
            f'Cannot parse YAML file {yaml_file}, '
            f'since it is not a file'
        )

    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
