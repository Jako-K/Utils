"""
Description:
Helper functions that didn't really fit in anywhere else.
As i get more functions, maybe these can be clustered into their own file.
But for know they just live here, without any real connection.
"""

import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd
from . import type_check as _type_check
import itertools as _itertools
import random as _random
import math as _math
from . import colors as _colors


def pandas_standardize_df(df:_pd.DataFrame):
    """
    Function description:
    Standardize pandas DataFrame

    Example:
    >> pandas_standardize_df(pd.DataFrame(np.array([1,2,3,4])))
              0
    0 -1.161895
    1 -0.387298
    2  0.387298
    3  1.161895
    """

    _type_check.assert_type(df, _pd.DataFrame)
    df_standardized = (df - df.mean()) / df.std()
    assert _np.isclose(df_standardized.mean(), 0), "Expected mean(std) ~= 0"
    assert _np.isclose(df_standardized.std(), 1), "Expected std(std) ~= 1"
    return df_standardized


def get_grid_coordinates(rows: int, cols: int):
    """
    # Function description
    Calculate 2D coordinates for grid traversing. If unclear, the example below should alleviate any doubt

    Example:
    >> get_grid_coordinates(3,2)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    @param rows: number of rows in grid
    @param cols: number of columns in grid
    """

    # Checks
    _type_check.assert_types([rows, cols], [int, int])
    _type_check.assert_comparison_number(rows, 1, ">=", "rows")
    _type_check.assert_comparison_number(cols, 1, ">=", "cols")

    return list(_itertools.product([i for i in range(rows)], [i for i in range(cols)]))


def sturges_rule(data):
    """
    Function description:
    Uses Sturges' rule to calculate bin_width and number_of_bins

    @param data: To be binned. `data` must have well defined: len, min and max

    Example:
    >> sturges_rule([1,2,3,4])
    (1.004420127756955, 3)
    """
    # NOTE Not sure about the intuition for the method or even how well it works, but
    # it seems like a reasonable way of picking bin sizes (and therefore #bins)

    try:
        len(data), min(data), max(data)
    except TypeError:
        raise TypeError("Expected `data` to have well defined `len`, `min` and `max` functions, but it didn't")

    k = 1 + 3.3 * _np.log10(len(data))
    bin_width = (max(data) - min(data)) / k
    number_of_bins = int(_np.round(1 + _np.log2(len(data))))

    return bin_width, number_of_bins


def unfair_coin_flip(p: float):
    """
    Function description:
    Flip a weighted coin

    @param p: Percentage of success should be range (0, 1)
    """
    _type_check.assert_type(p, float)
    if not (0.0<p<1.0):
        raise ValueError(f"0<p<1, but received p: `{p}`")
    return _random.random() > p


def int_sign(x: int):
    """ Return -1 if `x` is negative and 1 if `x` is positive """
    _type_check.assert_type(x, int)
    return _math.copysign(1, x)


def init_2d_list(rows:int, cols:int, value=None):
    """
    # Function description
    Construct a 2D list with dimensions `rows` x `cols` filled with `value`

    Example:
    >> init_2d_array(4,3)
    [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
    """
    return [[value for _ in range(cols)] for _ in range(rows)]


def ndarray_to_bins(array:_np.ndarray, num_bins:int = None):
    """
    Function description:
    Bin `array` into `num_bins` number of bins.

    @param array: Numpy array containing the values which is going to be binned
    @param num_bins: The number of bins used. If bins is None Sturges rule will be used automatically
    @return: array_binned (`array` binned), num_bins (the number of bins), thresholds (Thresholds used to bin)

    Example:
    >> ndarray_to_bins(np.array([1,2,3,4]), 2)
    (array([1, 1, 2, 3], dtype=int64), 2, array([1. , 2.5, 4. ]))
    """
    _type_check.assert_types([array, num_bins], [_np.ndarray, int], [0, 1])
    if num_bins is not None:
        _type_check.assert_comparison_number(num_bins, 1, ">=", "num_bins")

    if num_bins is None:
        num_bins = sturges_rule(array)[1]

    _, thresholds = _np.histogram(array, bins=num_bins)
    array_binned = _np.digitize(array, thresholds)
    return array_binned, num_bins, thresholds

__all__ = [
    "pandas_standardize_df",
    "get_grid_coordinates",
    "sturges_rule",
    "unfair_coin_flip",
    "int_sign",
    "init_2d_list",
    "ndarray_to_bins"
]


# TODO refactor, checks and remove underscore. Or maybe just delete it, it seems a bit to specific.
def _plot_average_uncertainty(data, stds=2):
    """
    data: np.array with shape (samples X repetitions)
    """
    xs = _np.arange(len(data))
    std = data.std(1)
    mean = data.mean(1)

    fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title("Individual")
    ax1.plot(data, ".-")

    ax2.set_title("Averaged with uncertainty")
    ax2.plot(mean, 'o-', color=_colors.color_from_name("blue", "hex"), label='Mean')
    _plt.sca(ax2)  # <-- makes gca work, super weird but gets the job done
    _plt.gca().fill_between(xs, mean - stds * std, mean + stds * std, color='lightblue', alpha=0.5,
                            label=r"$2\sigma$")
    _plt.plot(xs, [mean.mean()] * len(xs), '--', color=_colors.color_from_name("orange", "hex"),
              label="Mean of means")
    ax2.legend()
    _plt.show()

    return fig


