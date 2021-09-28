"""
Description
Stuff that hasen't been tested yet or that I'm on the fence about
"""

import pydicom as _dicom
import matplotlib.pylab as _plt
import numpy as _np
import warnings as _warnings

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


__all__ = [
    "show_dicom",
    "load_unspecified",
]