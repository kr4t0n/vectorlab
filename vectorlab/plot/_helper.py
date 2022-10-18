"""
These function is just matplotlib helper functions.
"""

import matplotlib.pyplot as plt


def show_plot():
    r"""Show existing matplotlib figure.
    """

    plt.show()

    return


def save_plot(file_path):
    r"""Save existing matplotlib figure to desired file_path.

    Parameters
    ----------
    file_path : str
        File path to save figure.
    """

    plt.savefig(file_path)

    return
