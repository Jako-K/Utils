"""
Description:
Help format stuff e.g. numbers in scientific notation
"""

from . import type_check as _type_check
import ast as _ast

def scientific_notation(number, num_mantissa:int):
    """ Rewrite number to scientific notation with `num_mantissa` amount of decimals """
    # Checks + int --> float cast if necessary
    if isinstance(number, int): number = float(number)
    _type_check.assert_types([number, num_mantissa], [float, int])

    return format(number, f".{num_mantissa}E")


def string_to_dict(string: str):
    """
    Function description:
    Convert string of format '{a:b...}' into a python dict

    Example:
    >> string_to_dict("{'a':1, 'b':2}")
    {'a': 1, 'b': 2}

    @param string: Python dictionary in string format
    """

    _type_check.assert_type(string, str)
    return _ast.literal_eval(string)


def string_to_list(string_list:str, element_type=None):
    """
    Function description:
    Convert string of format '[a, b...] into a python list

    EXAMPLE 1:
    >> string_to_list('[198, 86, 292, 149]')
    ['198', '86', '292', '149']

    EXAMPLE 2:
    >> string_to_list('[198, 86, 292, 149]', element_type=int)
    [198, 86, 292, 149]

    @param string_list: list in string representation
    @param element_type: automatic casting e.g. '[float, float]' --> [int, int]
    """
    # Tests
    _type_check.assert_types([string_list, element_type], [str, object], [0, 1])

    to_list = string_list.strip('][').split(', ')
    if element_type:
        to_list = list(map(element_type, to_list))
    return to_list

__all__ = [
    "scientific_notation",
    "string_to_dict",
    "string_to_list"
]