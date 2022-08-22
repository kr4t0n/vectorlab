"""
Some pre-defined, self-made palettes, and a function help
to load these palettes.
"""

import matplotlib.colors

from ..utils._check import check_valid_option

_palettes = {
    'black': ['#000000'],
    'default': [
        (0.24, 0.28, 0.37), (0.36, 0.65, 0.7),
        (0.5, 0.5, 0.5), (0.71, 0.40, 0.44)
    ],
    'high_contrast': [
        (0.18, 0.24, 0.30), (0.3, 0.5, 0.55), (0.65, 0.3, 0.35)
    ],
    'pale': [
        (180, 101, 111), (134, 187, 216),
        (148, 131, 146), (204, 252, 203)
    ],
    'fifth_dimension': [
        '#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'
    ],
    'rainbow': [
        '#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f',
        '#90be6d', '#43aa8b', '#4d908e', '#577590', '#277da1'
    ],
    'long': [
        '#22223b', '#4a4e69', '#9a8c98',
        '#999fa1', '#97b2aa', '#c9ada7',
        '#f2e9e4'
    ],
    'long_blue': [
        '#012a4a', '#013a63', '#01497c', '#014f86', '#2a6f97',
        '#2c7da0', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5'
    ],
    'long_contrast': [
        '#000000', '#7F95D1', '#FF82A9', '#FFC0BE', '#FFEBE7',
        '#A44200', '#D58936', '#F2F3AE', '#C4F1BE', '#A2C3A4',
        '#F4E76E', '#119822', '#2F4858', '#E3170A'
    ]
}


def _convert_color(color):
    r"""Convert a color into a tuple of RGB values.

    Parameters
    ----------
    color : str, tuple
        A hex string of a color, or a tuple of RGB of a color.

    Returns
    -------
    rgb : tuple
        A tuple of RGB values.
    """

    rgb = (0, 0, 0)

    if isinstance(color, str):
        rgb = matplotlib.colors.to_rgb(color)
    elif isinstance(color, tuple):
        rgb = tuple(
            element / 255 if isinstance(element, int) else element
            for element in color
        )

    return rgb


def load_palette(palette):
    r"""Load proper palette from given palette name or list of colors.

    Parameters
    ----------
    palette : str, list
        A palette name or a list of colors.

    Returns
    -------
    converted_palette : list
        A list of colors in coverted RGB tuple to form a palette.

    Raises
    ------
    ValueError
        If palette name is not defined, or palette is not a list of
        colors, a ValueError is raised.
    """

    if isinstance(palette, str):
        # treat palette as a palette name
        palette = check_valid_option(
            palette,
            options=list(_palettes.keys()),
            variable_name='palette name'
        )

        # load palette from pre-defined palettes
        palette = _palettes[palette]

    elif not isinstance(palette, list):

        raise ValueError(
            f'Palette shoud be a palette name or a self-defined '
            f'list of colors. Your palette type is {type(palette)}.'
        )

    # convert colors inside palette into desired format
    converted_palette = [
        _convert_color(color)
        for color in palette
    ]

    return converted_palette
