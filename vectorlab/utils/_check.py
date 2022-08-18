"""
Check input argument satisfy certain constraints.
"""

import torch
import numpy as np

from sklearn.metrics.pairwise import _return_float_dtype


def _check_ndarray(arr):
    r"""Check if input arr is a numpy ndarray

    Check whether the input `arr` is a numpy ndarray. If that so, return
    the input `arr`, otherwise, cast the input `arr` to numpy ndarray
    and then return it.

    Parameters
    ----------
    arr : list, array_like, shape (n, m)
        The input array.

    Returns
    -------
    np.ndarray
        The numpy ndarray version of input.
    """

    if not isinstance(arr, np.ndarray):
        return np.array(arr)

    return arr


def _check_tensor(arr):
    r"""Check if input arr is a torch tensor

    Check whether the input `arr` is a torch tensor. If that so, return
    the input `arr`, otherwise, cast the input `arr` to torch tensor
    and then return it.

    Parameters
    ----------
    arr : list, torch.tensor, shape (n, m)
        The input array.

    Returns
    -------
    torch.tensor
        The torch tensor version of input.
    """

    if not isinstance(arr, torch.Tensor):
        return torch.tensor(arr)

    return arr


def check_nd_array(X, n, dtype=None):
    r"""Set X appropriately and checks input

    Specifically, this function first ensures that X is an array,
    then checks that it is an nd array while ensuring that its elements
    are floats (or dtype if provided).

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples)
        The input data.
    n : int
        The number of dimensions.
    dtype : str, type, list of types, optional
        Data type required for X. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples)
        An array equal to X, guaranteed to be a numpy array.

    Raises
    ------
    ValueError
        When X is not a nd array, a ValueError is raised.
    """

    X, _, dtype_float = _return_float_dtype(X, None)

    if dtype is None:
        dtype = dtype_float

    X = X.astype(dtype=dtype)

    if X.ndim != n:
        raise ValueError(
            f'Invalid dimension for X: '
            f'X.shape == {X.shape}, it should be a {n}d array'
        )

    return X


def check_pairwise_1d_array(X, Y, dtype=None):
    r"""Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy)
    If Y is given, this does not happen.
    All 1d distance metrics should use this function first to assert
    that given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are 1d array while ensuring that their elements
    are floats (or dtype if provided).

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples)
        The first input data.
    Y : {array-like, sparse matrix}, shape (n_samples)
        The second inputs data.
    dtype : str, type, list of types, optional
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples)
        An array equal to X, guaranteed to be a numpy array.
    safe_Y : {array-like, sparse matrix}, shape (n_samples)
        An array equal to Y if Y was not None,
        guaranteed to be a numpy array. If Y was None,
        safe_Y will be a pointer to X.

    Raises
    ------
    ValueError
        When X and Y have different dimension or length,
        a ValueError is raised.
    """

    X, Y, dtype_float = _return_float_dtype(X, Y)

    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        Y = X = X.astype(dtype=dtype)

    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError(
            f'Invalid dimension for X and Y: '
            f'X.shape == {X.shape} while Y.shape == {Y.shape}, '
            f'they should be 1d array'.format
        )
    elif X.shape[0] != Y.shape[0]:
        raise ValueError(
            f'Incompatible length for X and Y '
            f'X.shape[0] == {X.shape[0]} while Y.shape[0] == {Y.shape[0]}'
        )

    return X, Y


def check_valid_int(x,
                    lower=None, upper=None,
                    lower_inclusive=True,
                    upper_inclusive=True,
                    variable_name=None):
    r"""Check int type of a variable and certain constraint

    Check whether a variable `x` is an integer. Certain constraint
    can be given to check whether such variable satisfying or not.
    When lower and upper constraint is None, it will be set to `-inf`
    and `inf` respectfully. `lower_inclusive` and `upper_inclusive`
    suggests whether it's a constraint of `(,)` or `[,]`. A variable
    name can also be provided to make the error message more informative.

    Parameters
    ----------
    x : object
        The variable object to be checked.
    lower : int, optional
        The lower constraint.
    upper : int, optional
        The upper constraint.
    lower_inclusive : bool, optional
        Include lower constraint or not.
    upper_inclusive : bool, optional
        Include upper constraint or not.
    variable_name : str, optional
        The variable name to be announced.

    Returns
    -------
    x : int
        The integer type of x if possible.

    Raises
    ------
    ValueError
        If the parameter `x` cannot be treated as an integer or
        the parameter `x` does not satisfy the constraint, a
        ValueError is raised.
    """

    lower = lower if lower is not None else -np.Inf
    upper = upper if upper is not None else np.Inf

    if isinstance(x, float) and int(x) == x:
        x = int(x)

    if isinstance(x, int) or isinstance(x, np.int_):
        lower_check = (lower <= x if lower_inclusive else lower < x)
        upper_check = (x <= upper if upper_inclusive else x < upper)

        if lower_check and upper_check:
            return x

    if variable_name:
        variable_name += ' '
    else:
        variable_name = ''

    lower_clause = '[' if lower_inclusive else '('
    upper_clause = ']' if upper_inclusive else ')'

    raise ValueError(
        f'Variable {variable_name}does not match the constraint. '
        f'Target type is integer, your type is {type(x)}. '
        f'The value should between {lower_clause}{lower}, '
        f'{upper}{upper_clause}, your value is {x}.'
    )


def check_valid_float(x,
                      lower=None, upper=None,
                      lower_inclusive=True,
                      upper_inclusive=True,
                      variable_name=None):
    r"""Check float type of a variable and certain constraint

    Check whether a variable `x` is a float. Certain constraint
    can be given to check whether such variable satisfying or not.
    When lower and upper constraint is None, it will be set to `-inf`
    and `inf` respectfully. `lower_inclusive` and `upper_inclusive`
    suggests whether it's a constraint of `(,)` or `[,]`. A variable
    name can also be provided to make the error message more informative.

    Parameters
    ----------
    x : object
        The variable object to be checked.
    lower : float, optional
        The lower constraint.
    upper : float, optional
        The upper constraint.
    lower_inclusive : bool, optional
        Include lower constraint or not.
    upper_inclusive : bool, optional
        Include upper constraint or not.
    variable_name : str, optional
        The variable name to be announced.

    Returns
    -------
    x : float
        The float type of x if possible.

    Raises
    ------
    ValueError
        If the parameter `x` cannot be treated as a float or
        the parameter `x` does not satisfy the constraint, a
        ValueError is raised.
    """

    lower = lower if lower is not None else -np.Inf
    upper = upper if upper is not None else np.Inf

    if isinstance(x, int):
        x = float(x)

    if isinstance(x, float) or isinstance(x, np.float_):
        lower_check = (lower <= x if lower_inclusive else lower < x)
        upper_check = (x <= upper if upper_inclusive else x < upper)

        if lower_check and upper_check:
            return x

    if variable_name:
        variable_name += ' '
    else:
        variable_name = ''

    lower_clause = '[' if lower_inclusive else '('
    upper_clause = ']' if upper_inclusive else ')'

    raise ValueError(
        f'Variable {variable_name}does not match the constraint. '
        f'Target type is float, your type is {type(x)}. '
        f'The value should between {lower_clause}{lower}, '
        f'{upper}{upper_clause}, your value is {x}.'
    )


def check_valid_option(x, options,
                       variable_name=None):
    r"""Check option and certain constraint

    Check whether a variable `x` is a valid option. Certain constraint
    can be given to check whether such variable is one of them. A variable
    name can also be provided to make the error message more informative.

    Parameters
    ----------
    x : object
        The variable object to be checked.
    options : list
        List of potential valid options.
    variable_name : str, optional
        The variable name to be announced.

    Returns
    -------
    x : object
        The valid option x if possible.

    Raises
    ------
    ValueError
        if the parameter `x` is not one the provided valid options,
        a ValueError is raised.
    """

    if x in options:
        return x

    if variable_name:
        variable_name += ' '
    else:
        variable_name = ''

    raise ValueError(
        f'Variable {variable_name}does not match the constraint. '
        f'Potential options are {options}, your option is {x}.'
    )
