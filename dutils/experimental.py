"""
Description
Stuff that hasen't been tested yet or that I'm on the fence about
"""
import pydicom as _dicom
import matplotlib.pylab as _plt
import numpy as _np
import warnings as _warnings
import pandas as _pd
import seaborn as _sns


from . import colors as _colors
from . import input_output as _input_output
from . import type_check as _type_check
from . import images as _images


def show_dicom(path:str):
    # Checks
    _input_output.assert_path(path)
    if _input_output.get_file_extension(path)[-4:] != ".dcm":
        raise ValueError("Expected `.dcom` extension, but received something else")
    
    dicom_image = _dicom.dcmread(path).pixel_array
    _plt.imshow(dicom_image, cmap="bone")
    _plt.axis("off")


def load_unspecified(path:str):
    """
    Try and load whatever is being passed: image.jpg, sound.wav, text.txt etc.
    and return it in an appropriote format.

    # TODO is this a good idea?
    
    # image [.jpg, .png] as nd.array
    # dicom [.dcm] as nd.array
    # text [.txt, .json] as string
    # sound [wav] as nd.array
    # video [mp4, avi] as list[nd.array images]

    """
    raise NotImplementedError("")


def confusion_matrix_binary(targets:_np.ndarray, preds:_np.ndarray, plot:bool=True):
    """
    Plot a classic confusion matrix for binary `targets` and `predictions` alongside som key statistics.

    EXAMPLE:
    >> t = np.array([1,0,1,0,0,0,1,0,0,1,1,0])
    >> p = np.array([1,0,1,1,0,1,1,0,1,0,0,1])
    >> U.experimental.confusion_matrix_binary(t,p)

    @param targets: A np.ndarray containing the real values
    @param preds: A np.ndarray containing the prediction values
    @param plot: If the plt.show() should be called within the function
    @return: (fig: plt figure, ax: plt axis, key_values: dict with key statistics e.g. accuracy).
    """

    # Checks
    _type_check.assert_types([targets, preds, plot], [_np.ndarray, _np.ndarray, bool])
    if len(targets) == 1 and len(preds) == 1:
        raise ValueError("Expected `target` and `preds` to be 1 dimensional, "
                         f"but received `{targets.shape}` and `{preds.shape}`")
    if targets.shape[0] != preds.shape[0]:
        raise ValueError(f"Length mismatch. `len(targets)={len(targets)}` and `len(preds)={len(preds)}`")
    if (targets.dtype.kind not in list('buif')) or (preds.dtype.kind not in list('buif')):
        raise TypeError("`targets` and/or `preds` contain non-numerical values")
    if _np.in1d(targets, [0, 1]).sum() != targets.shape[0]:
        raise ValueError("Expected `targets` to only contain 0 and 1, but received something else")
    if _np.in1d(preds, [0, 1]).sum() != preds.shape[0]:
        raise ValueError("Expected `preds` to only contain 0 and 1, but received something else")


    # Construct confusion matrix and its plt stuff
    cm = _pd.crosstab(
        _pd.Series(targets, name="Actual"),
        _pd.Series(preds, name="Predicted"),
        rownames=['Actual'], colnames=['Predicted']
    )
    fig, ax = _plt.subplots(figsize=(10,8))
    _sns.heatmap(cm / cm.sum().sum(), annot=True, cmap="Blues", ax=ax)

    # Calculate key stats
    key_values = {
        "accuracy": (targets == preds).sum() / len(targets),
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
    for (x,y,t) in [(0.43, 0.4,"TN"), (1.43, 0.4,"FP"), (0.43, 1.4,"FN"), (1.43, 1.4,"TP")]:
        ax.text(x, y, t, fontsize=15, color=(0.25,0.25,0.25), bbox={'facecolor': 'lightblue', 'alpha':0.5})

    if plot: _plt.show()
    return fig, ax, key_values


__all__ = [
    "show_dicom",
    "load_unspecified",
    "confusion_matrix_binary",
]