"""
Description:
Helper functions that didn't really fit in anywhere else.
As i get more functions, maybe these can be clustered into their own file.
But for know they just live here, without any real connection.
"""

import numpy as _np
import pandas as _pd
import itertools as _itertools
import random as _random
import math as _math
import seaborn as _sns
import matplotlib.pylab as _plt

from . import type_check as _type_check


def scientific_notation(number, num_mantissa:int):
    """ Rewrite `number` to scientific notation with `num_mantissa` amount of decimals """
    # Checks + int --> float cast if necessary
    if isinstance(number, int): number = float(number)
    _type_check.assert_types([number, num_mantissa], [float, int])

    return format(number, f".{num_mantissa}E")

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
    Calculate 2D coordinates for grid traversing.

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
    # NOTE Don't understand the motivation behind the method or even how well it works, but
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
    return int(_math.copysign(1, x))


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


# TODO Unit tests
def confusion_matrix_binary(targets:_np.ndarray, preds:_np.ndarray, plot:bool=True):
    """
    Plot a classic confusion matrix for binary `targets` and `predictions` alongside som key statistics.

    EXAMPLE:
    >> t = np.array([1,0,1,0,0,0,1,0,0,1,1,0])
    >> p = np.array([1,0,1,1,0,1,1,0,1,0,0,1])
    >> U.all_around.confusion_matrix_binary(t,p)

    @param targets: A np.ndarray containing the real values
    @param preds: A np.ndarray containing the prediction values
    @param plot: If the plt.show() should be called within the function
    @return: (fig: plt figure, ax: plt axis, key_values: dict with key statistics e.g. accuracy).
    """

    # Checks
    _type_check.assert_types([targets, preds, plot], [_np.ndarray, _np.ndarray, bool])
    if len(targets) == 1 and len(preds) == 1:
        raise ValueError("Expected `targets` and `preds` to be 1 dimensional, "
                         f"but received `{targets.shape}` and `{preds.shape}`")
    if targets.shape[0] != preds.shape[0]:
        raise ValueError(f"Length mismatch. `len(targets)={len(targets)}` and `len(preds)={len(preds)}`")
    if (targets.dtype.kind not in list('buif')) or (preds.dtype.kind not in list('buif')):
        raise TypeError("`targets` and/or `preds` contain non-numerical values")
    if _np.in1d(targets, [0, 1]).sum() != targets.shape[0]:
        raise ValueError("Expected `targets` to only contain 0 and 1, but received something else")
    if _np.in1d(preds, [0, 1]).sum() != preds.shape[0]:
        raise ValueError("Expected `preds` to only contain 0 and 1, but received something else")
    if preds.sum() == len(preds) or targets.sum() == len(targets) or not preds.sum() or not targets.sum():
        raise ValueError("`targets` and `preds` must both contain at least one `0` and one `1`")


    # Construct confusion matrix and its plt stuff
    cm = _pd.crosstab(
        _pd.Series(targets, name="Actual"),
        _pd.Series(preds, name="Predicted"),
        rownames=['Actual'], colnames=['Predicted']
    )
    fig, ax = _plt.subplots(figsize=(10,8))
    _sns.heatmap(cm / cm.sum().sum(), annot=True, cmap="Blues", annot_kws={"size":25}, ax=ax)

    # Calculate key stats
    key_values = {
        "accuracy": round((targets == preds).sum() / len(targets),3),
        "sensitivity": round(cm.loc[1, 1] / (cm.loc[1, 1] + cm.loc[0, 1]), 3),
        "specificity": round(cm.loc[0, 0] / (cm.loc[0, 0] + cm.loc[1, 0]), 3),
        "0/1 target balance": (round(targets.sum()/len(targets),3), round(1 - targets.sum()/len(targets),3))
    }

    # Add key stats to the title
    to_title = ""
    for name, value in key_values.items():
        to_title += f"{name}: {value}  |  "
    ax.set_title(to_title[:-4])

    # Add labels (e.g. FN = False negative) to each cell
    for (x,y,t) in [(0.425, 0.35,"TN"), (1.425, 0.35,"FP"), (0.425, 1.35,"FN"), (1.425, 1.35,"TP")]:
        ax.text(x, y, t, fontsize=24, color=(0.25,0.25,0.25), bbox={'facecolor': 'lightblue', 'alpha':0.5})

    if plot: _plt.show()
    return fig, ax, key_values


__all__ = [
    "scientific_notation",
    "pandas_standardize_df",
    "get_grid_coordinates",
    "sturges_rule",
    "unfair_coin_flip",
    "int_sign",
    "init_2d_list",
    "ndarray_to_bins",
    "confusion_matrix_binary"
]
