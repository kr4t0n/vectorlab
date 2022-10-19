def _is_notebook():
    r"""Determine the whether the code is executed inside notebook
    of python CLI.

    Returns
    -------
    bool
        Whether the code is executed inside Jupyter notebook.
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
