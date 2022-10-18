"""
This module provides a loading decorator for time
consuming functions.
"""

import time

from functools import wraps

from ._env import _is_notebook

if _is_notebook():
    from halo import HaloNotebook as Halo
else:
    from halo import Halo


def loading(text='Loading ...',
            spinner_style='dots',
            success_text='Success',
            failed_text='Failed',
            with_time=True):
    r"""The loading decorator for spinning and timing.

    This decorator could create a spinner for more interactive
    terminal execution. This decorator could also for timing
    purpose.

    Parameters
    ----------
    text : str, optional
        The text showed when inner function is executing.
    spinner_style : str, optional
        The spinner style to show when inner function is executing.
    success_text : str, optional
        The text showed when inner function is succeed.
    failed_text : str, optional
        The text showed when inner function is failed.
    with_time : bool, optional
        Whether turn timing function on.

    Returns
    -------
    decorator
        The inner actual decorator without arguments.
    """

    def loading_decorator(func):

        @ wraps(func)
        def wrapper(*args, **kwargs):

            with Halo(text=text, spinner=spinner_style) as spinner:
                try:
                    if with_time:
                        ts = time.time()

                    result = func(*args, **kwargs)

                    if with_time:
                        te = time.time()
                        spinner.succeed(
                            f'{success_text} [time: {te - ts:4.4f}s]'
                        )
                    else:
                        spinner.succeed(success_text)
                except Exception as e:
                    spinner.fail(
                        f'{failed_text} {e}'
                    )
                    raise e

            return result

        return wrapper

    return loading_decorator
