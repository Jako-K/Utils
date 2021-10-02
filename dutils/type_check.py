"""
DESCRIPTION:
Make it more convenient to check types
"""
# TODO: Add a is_iterable function: which would for instance accept both: list, tuples, np.ndarray etc.
# TODO: Add `assert_in` function which can check if a value is in a list/tuple
# TODO Fix the order: assert_comparison_number(rows, 1, ">=", "rows") ==> rows, ">=", 1, "rows".

from types import ModuleType
NoneType = type(None)


# General
def assert_type(to_check, expected_type, allow_none:bool=False):
    """
    Function description:
    Check object against expected type

    @param to_check: Object for type check
    @param expected_type: Expected type of `to_check`
    @param allow_none: Weather or not None is an accepted type or not
    """

    if not isinstance(allow_none, bool):
        raise ValueError(f"Expected `allow_None` to by of type bool, but received type `{type(allow_none)}`")
    if (to_check is None) and (expected_type is None):
        raise TypeError(f"`None` is not a valid type. If you're trying to check if `type(to_check) == None` try set"
                        f" `expected_type=type(None)` instead.")

    is_ok = isinstance(to_check, expected_type)
    if allow_none:
        is_ok = (to_check is None) or is_ok

    if not is_ok:
        raise TypeError(f"Expected type `{expected_type}`, but received type `{type(to_check)}`")


def assert_types(to_check:list, expected_types:list, allow_nones:list=None):
    """
    Function description:
    Check list of values against expected types

    @param to_check: List of values for type check
    @param expected_types: Expected types of `to_check`
    @param allow_nones: list of booleans or 0/1
    """

    # Checks
    assert_type(to_check, list)
    assert_type(expected_types, list)
    assert_type(allow_nones, list, allow_none=True)
    if len(to_check) != len(expected_types):
        raise ValueError("length mismatch between `to_check_values` and `expected_types`")

    # If `allow_nones` is None all values are set to False.
    if allow_nones is None:
        allow_nones = [False for _ in range(len(to_check))]
    else:
        if len(allow_nones) != len(to_check):
            raise ValueError("length mismatch between `to_check_values` and `allow_nones`")
        for i, element in enumerate(allow_nones):
            if element in [0, 1]:
                allow_nones[i] = element == 1 # the `== 1` is just to allow for zeros as False and ones as True

    # check if all elements are of the correct type
    for i, value in enumerate(to_check):
        assert_type(value, expected_types[i], allow_nones[i])


def assert_list_slow(to_check:list, expected_type, expected_length:int=None, allow_none:bool=False):
    """
    Function description:
    Check the values of `to_check` against `expected_type` and `expected_length`.

    @param to_check: List of values for type check
    @param expected_type: Expected type of every element in `to_check`
    @param expected_length: Expected length of `to_check`
    @param allow_none: Weather or not None is an accepted type
    """
    # Tests
    assert_types([to_check, expected_length, allow_none], [list, int, bool], [0, 1, 0])
    if expected_length is None:
        pass
    elif expected_length < 0:
        raise ValueError(f"`expected_length >= 0, but received `{expected_length}`")
    elif len(to_check) != expected_length:
        raise ValueError(f"Expected length `{expected_length}`, but received length `{len(to_check)}`")


    # check if all elements are of the correct type
    for element in to_check:
        if not isinstance(element, expected_type) and not (isinstance(element, NoneType) if allow_none else 0):
            raise TypeError(f"Found element of type `{type(element)}`, but expected `{expected_type}`")


def assert_in(to_check, check_in):
    """
    Function description:
    Check if the value `to_check` is present in `check_in`

    @param to_check: Value to be checked
    @param check_in: Values `to_check` is being checked against
    """
    try:
        is_in = to_check in check_in
    except Exception:
        raise RuntimeError(f"Failed to execute `to_check in check_in`")

    if not is_in:
        raise ValueError(f"Expected `{to_check}` to be in `{check_in}`, but it wasn't")


def assert_comparison_number(number, check_against, comparison:str, number_name:str):
    """
    Function description:
    Check if `number` compare correctly against `check_against`

    Example:
    >> assert_comparison_number(3, 0, "<=", "number_of_cats")
    ValueError(...)
    >> assert_comparison_number(3, 0, ">=", "number_of_cats")
    None

    @param number: Value to be checked
    @param check_against: Value `number` is checked against
    @param comparison: Comparison operator, must be in [">", "<", "<=", ">=", "=", "=="]
    @param number_name: The name of `number`which is used in the error prints

    """

    # Checks
    assert_type(number, (float, int))
    assert_types([comparison, number_name], [str, str])
    assert_in(comparison, [">", "<", "<=", ">=", "=", "=="])
    if type(number) != type(check_against):
        raise TypeError(f"Type mismatch between `number` and `check_against`, "
                         f"received type {type(number)} and {type(check_against)} respectively")

    less, higher, equal = (comparison == "<"), (comparison == ">"), (comparison in ["=", "=="])
    less_or_equal, higher_or_equal = (comparison == "<="), (comparison == ">=")

    if equal and not (number == check_against):
        raise ValueError(f"Expected `{number_name}` == `{check_against}`, but received `{number_name}={number}`")
    elif less and not (number < check_against):
        raise ValueError(f"Expected `{number_name}` < `{check_against}`, but received `{number_name}={number}`")
    elif less_or_equal and not (number <= check_against):
        raise ValueError(f"Expected `{number_name}` <= `{check_against}`, but received `{number_name}={number}`")
    elif higher and not (number > check_against):
        raise ValueError(f"Expected `{number_name}` > `{check_against}`, but received `{number_name}={number}`")
    elif higher_or_equal and not (number >= check_against):
        raise ValueError(f"Expected `{number_name}` >= `{check_against}`, but received `{number_name}={number}`")



__all__ = [
    "assert_type",
    "assert_types",
    "assert_list_slow",
    "assert_in",
    "assert_comparison_number",
    "NoneType",
    "ModuleType",
]
