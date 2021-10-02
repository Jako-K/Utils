"""
Description:
Helper functions for working with colors.
Intended use through the class instance `colors`.

Example:
>> colors.random_color(color_type="hex", amount=2)
['#c51cbe', '#0dc76a']
"""
# TODO: Add support for BGR
# TODO: Add @param to all

import re as _re
import random as _random
from PIL import ImageColor as _ImageColor
import numpy as _np
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt

from . import type_check as _type_check

# Seaborn color scheme
_seaborn_blue = (31, 119, 180)
_seaborn_orange = (255, 127, 14)
_seaborn_green = (44, 160, 44)
_seaborn_red = (214, 39, 40)
_seaborn_purple = (148, 103, 189)
_seaborn_brown = (140, 86, 75)
_seaborn_pink = (227, 119, 194)
_seaborn_grey = (127, 127, 127)
_seaborn_white = (225, 255, 255)
_seaborn_colors = {"blue": _seaborn_blue,
                  "orange": _seaborn_orange,
                  "green": _seaborn_green,
                  "red": _seaborn_red,
                  "purple": _seaborn_purple,
                  "brown": _seaborn_brown,
                  "pink": _seaborn_pink,
                  "grey": _seaborn_grey,
                  "white": _seaborn_white}


_legal_types = ["rgb", "rgb_01", "hex"]
_scheme_name_to_colors = {"seaborn": _seaborn_colors}
_colors_schemes = list(_scheme_name_to_colors.keys())


def is_legal_hex(color: str):
    return isinstance(color, str) and (_re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color) is not None)


def is_legal_rgb(color):
    """ Legal RGB colors are of type tuple/list and comprised of 3 integers in the range 0-255 """
    if not isinstance(color, (tuple, list)):  # Is a list or a tuple
        return False
    if len(color) != 3:  # Has len 3
        return False
    if sum([isinstance(color_channel, int) for color_channel in color]) != 3:  # All channels is of type int
        return False
    if sum([0 <= color_channel <= 255 for color_channel in color]) != 3:  # All channels is within 0-256
        return False

    return True


# TODO Update unit tests to include "rgb_01"
def is_legal_rgb_01(color):
    """ Legal RGB colors are of type tuple/list and comprised of 3 floats in the range 0-1 """
    if not isinstance(color, (tuple, list)):  # Is a list or a tuple
        return False
    if len(color) != 3:  # Has len 3
        return False
    if sum([isinstance(color_channel, float) for color_channel in color]) != 3:  # All channels is of type float
        return False
    if sum([0.0 <= color_channel <= 1.0 for color_channel in color]) != 3:  # All channels is within 0-1
        return False

    return True


# TODO Update unit tests to include "rgb_01"
def get_color_type(color):
    """ Try to detect and return color type, only `legal_types` color types are supported """
    if is_legal_hex(color):
        return "hex"
    elif is_legal_rgb(color):
        return "rgb"
    elif is_legal_rgb_01(color):
        return "rgb_01"

    return None


def _assert_type_str(color_type: str):
    """ assert color type is supported """
    _type_check.assert_type(color_type, str)
    if color_type not in _legal_types:
        raise ValueError(f"Received unknown color type `{color_type}`. Legal types are: `{_legal_types}`")


def assert_color(color):
    """ Detect color type and assert it's supported """
    if get_color_type(color) is None:
        raise TypeError(f"Color format cannot be interpreted. Legal color types are: `{_legal_types}`")


def _assert_color_scheme(scheme: str):
    """ assert color scheme is supported """
    _type_check.assert_type(scheme, str)
    if scheme not in _colors_schemes:
        raise ValueError(f"Received unknown color scheme `{scheme}`. Legal types: `{_colors_schemes}`")


def _assert_color_word(color_name:str, scheme_name:str):
    """ Check if `color_name` is in `scheme_name` """
    # Checks
    _type_check.assert_types([color_name, scheme_name], [str, str])
    _assert_color_scheme(scheme_name)

    color_scheme = _scheme_name_to_colors[scheme_name]
    legal_colors = list(color_scheme.keys())
    if color_name not in legal_colors:
        raise ValueError(f"The color `{color_name}` is not present in the color scheme `{scheme_name}`.\n"
                         f" Legal colors in `{scheme_name}`: {legal_colors}")


# TODO Update unit tests to include "rgb_01"
def convert_color(color, convert_to:str):
    """ convert color from one format to another e.g. from RGB --> HEX """
    _type_check.assert_type(convert_to, str)
    _assert_type_str(convert_to)
    assert_color(color)
    convert_from_type = get_color_type(color)

    # Going to convert `color` to RGB format regardless of `convert_to` and `convert_from_type`.
    # It's necessary to know `convert_from_type` with certainty to avoid a factorial number of if statements
    # That is, avoid stuff like --> if (convert_from_type == "hex") and (convert_to == "rgb"): ...
    if convert_from_type != "rgb":
        if convert_from_type == "hex": color = _hex_to_rgb(color)
        elif convert_from_type == "rgb_01": [int(c*255) for c in color] # rgb_01 -> rgb
        else: raise AssertionError("Shouldn't have gotten this far")


    if (convert_from_type == convert_to) or (convert_to == "rgb"):
        return color
    elif convert_to == "hex":
        return _rgb_to_hex(color)
    elif convert_to == "rgb_01":
        return [round(c/255.0,3) for c in color]
    else:
        raise AssertionError("Shouldn't have gotten this far")


def random_color(amount:int=1, color_type:str="rgb", min_rgb:int=0, max_rgb:int=255):
    """
    return `amount` number of random colors in accordance with `min_rgb` and `max_rgb`
    in the color format specified by `color_type`.
    """
    _type_check.assert_types([amount, color_type, min_rgb, max_rgb], [int, str, int, int])
    _assert_type_str(color_type)

    if not (0 <= min_rgb <= 255):
        raise ValueError("Expected min_rgb in 0-255, received {min_rgb}")
    if not (0 <= max_rgb <= 255):
        raise ValueError("Expected max_rgb in 0-255, received {max_rgb}")
    if max_rgb <= min_rgb:
        raise ValueError("Received min_rgb > max_rgb")
    if amount < 1:
        raise ValueError("Received amount < 1")

    generated_colors = []
    for _ in range(amount):
        color = [_random.randint(min_rgb, max_rgb) for _ in range(3)]
        color_converted = convert_color(color, color_type)
        generated_colors.append(color_converted)

    return generated_colors[0] if (amount == 1) else generated_colors


def _hex_to_rgb(hex_color: str):
    _type_check.assert_type(hex_color, str)
    if not is_legal_hex(hex_color):
        raise ValueError(f"`hex_color={hex_color}` is not recognized as a HEX color")
    return _ImageColor.getcolor(hex_color, "RGB")


def _rgb_to_hex(rgb_color):
    _type_check.assert_type(rgb_color, (tuple, list))
    if not is_legal_rgb(rgb_color):
        raise ValueError(f"`rgb_color={rgb_color}` is not recognized as a RGB color")
    return "#" + '%02x%02x%02x' % tuple(rgb_color)


def get_color(color_name:str, color_type:str= "rgb", color_scheme:str= "seaborn"):
    """
    Return color of name `color_name` from `color_scheme` in the format specified by `color_type`.
    Note: `color_name` should only contain the acutal color e.g. "blue" without any prefix e.g. "seaborn_blue"
    """

    _type_check.assert_types([color_name, color_type, color_scheme], [str, str, str])
    _assert_type_str(color_type)
    _assert_color_scheme(color_scheme)
    _assert_color_word(color_name, color_scheme)

    color_scheme = _scheme_name_to_colors[color_scheme]
    color = color_scheme[color_name]
    color_converted = convert_color(color, color_type)

    return color_converted


def display_colors(colors: list):
    """ Display all colors in `colors` in a matplotlib plot with corresponding hex, rgb etc. values"""
    # Checks
    _type_check.assert_type(colors, list)
    if len(colors) < 1:
        raise ValueError(f"Expected at least 1 color, received `{len(colors)}` number of colors")
    for color in colors:
        assert_color(color)

    # The plot expects colors of type RGB
    colors = [convert_color(color, "rgb") for color in colors]

    fig, ax = _plt.subplots(figsize=(15, len(colors)))
    _plt.xlim([0, 100])
    _plt.ylim([0, 100])
    square_height = 100 / len(colors)

    for i, color in enumerate(colors):
        assert_color(color)

        # matplotlib's Rectangle expect RGB channels in 0-1
        color_rgb = convert_color(color, "rgb")
        color_rgb_01 = [c / 255 for c in color_rgb]

        # Draw colored rectangles
        y_start = 100 - (i + 1) * square_height
        rect = _matplotlib.patches.Rectangle((0, y_start), 100, square_height, color=color_rgb_01)
        ax.add_patch(rect)

        # Write colors in all legal formats
        for j, color_type in enumerate(["rgb", "hex"]):
            color_text = convert_color(color, color_type)
            if color_type == "rgb":
                color_text = [" " * (3 - len(str(c))) + str(c) for c in color]
            text = f"{color_type}: {color_text}".replace("'", "")

            # White text if light color, black text if dark color + text plot
            brightness = _np.mean(color_rgb)
            text_color = "black" if brightness > 50 else "white"
            _plt.text(5 + j * 20, y_start + square_height // 2 - 0.5, text, color=text_color, size=15)

    _plt.axis("off")
    _plt.show()

    return fig, ax


def get_color_scheme(color_scheme:str, color_type:str= "rgb"):
    """ Return the color values from `color_scheme` in the format specified by `color_type`"""
    _type_check.assert_types([color_scheme, color_type], [str, str])
    _assert_type_str(color_type)
    _assert_color_scheme(color_scheme)

    # Grab all the color values and change their format to match that of `color_type`
    colors = _scheme_name_to_colors[color_scheme].values()
    return [convert_color(color, color_type) for color in colors]


def get_colors(colors: list, color_scheme="seaborn", color_type="rgb", detailed: bool = False):
    """
    Return every color:str in `colors` from `color_scheme` in format `color_type`.

    @param colors: List of colors in plain english e.g. "blue"
    @param color_scheme: Color scheme e.g. "seaborn"
    @param color_type: Format of colors e.g. "rgb"
    @param detailed: If true return a dict with name of color and value e.g. `{"blue":(31, 119, 180)}`.
                     If false, returns a list of values instead.
    @return: List of colors or dict with colors and their name.
    """

    # Checks
    _type_check.assert_types([colors, color_scheme, color_type, detailed], [list, str, str, bool])

    return_colors = {}
    for color in colors:
        return_colors[color] = get_color(color, color_type, color_scheme)

    return return_colors if detailed else list(return_colors.values())


__all__ = [
    "is_legal_hex",
    "is_legal_rgb",
    "get_color_type",
    "assert_color",
    "convert_color",
    "random_color",
    "get_color",
    "display_colors",
    "get_color_scheme",
    "get_colors",
    "is_legal_rgb_01",
]