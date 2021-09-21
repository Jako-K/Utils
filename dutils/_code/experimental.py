"""
Description
Stuff that hasen't been tested yet or that I'm on the fence about
"""

import pydicom as _dicom
import matplotlib.pylab as _plt
import numpy as _np

from . import colors as _colors
from . import input_output as _input_output


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


def plot_average_uncertainty(data, stds=2):
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
    ax2.plot(mean, 'o-', label='Mean')
    _plt.sca(ax2)  # <-- makes gca work, super wierd but gets the job done
    _plt.gca().fill_between(xs, mean - stds * std, mean + stds * std, color='lightblue', alpha=0.5, label=r"$2\sigma$")
    _plt.plot(xs, [mean.mean()] * len(xs), '--', color=_colors.seaborn_orange, label="Mean of means")
    ax2.legend()
    _plt.show()

    return fig


__all__ = [
    "show_dicom",
    "load_unspecified",
    "plot_average_uncertainty",
]