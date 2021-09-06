"""
Description
Stuff that hasen't been tested
"""


import pydicom as _dicom
import matplotlib.pylab as _plt
import numpy as _np
import cv2 as _cv2

from . import images as _images
from . import colors as _colors
from . import type_check as _type_check
from . import input_output as _input_output

def show_dicom(path:str):
    # Checks
    _input_output.assert_path(path)
    if _input_output.get_file_extension(path)[-4:] != ".dcm":
        raise ValueError("Expected `.dcom` extension, but recieved something else")
    
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


def get_colors(colors:list, color_scheme="seaborn", color_type="rgb"):
    scheme = _colors._scheme_name_to_colors[color_scheme]
    return [_colors.convert_color(scheme[color], color_type) for color in colors]


# These are supposed to be inside colors.convert put haven't gotten around to it, 
# so are just going to but them here for now
def BGR2RGB(image:_np.ndarray):
    return _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
def RGB2BGR(image:_np.ndarray):
    return _cv2.cvtColor(image, _cv2.COLOR_RGB2BGR)



def show_hist(image:_np.ndarray):
    """
    If image has 3 channels its expected to be RGB not BRG
    """
    _type_check.assert_type(image, _np.ndarray)
    _images.assert_ndarray_image(image)

    if len(image.shape) == 2:
        image = _np.expand_dims(image, axis=2)

    channels = image.shape[2]    
    if channels not in [1, 3]:
        raise ValueError("Expected `image` to have 1 or 3 channels, but recieved `{images.shape[2]}`")

    if channels == 3: #RGB image
        colors = get_colors(["red", "green", "blue"], "seaborn", "hex")
        fig, axes = _plt.subplots(3, 1, figsize=(15, 15))
        titles = ["Red", "Green", "Blue"]
    elif channels == 1:
        colors = ["grey"]
        fig, axes = _plt.subplots(figsize=(15,5))
        titles = ["Greyscale"]    

    for i in range(channels):
        ax = axes[i] if image.shape[2] > 1 else axes
        ax.hist(_np.ndarray.flatten(image[:,:,i]), color=colors[i], bins=255)
        ax.set_title(titles[i] + " Color Channel")
        ax.set_ylabel("Pixel count")
    ax.set_xlabel("Pixel value")


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
    ax2.plot(mean, 'o-', color=_colors.seaborn_blue, label='Mean')
    _plt.sca(ax2)  # <-- makes gca work, super wierd but gets the job done
    _plt.gca().fill_between(xs, mean - stds * std, mean + stds * std, color='lightblue', alpha=0.5, label=r"$2\sigma$")
    _plt.plot(xs, [mean.mean()] * len(xs), '--', color=_colors.seaborn_orange, label="Mean of means")
    ax2.legend()
    _plt.show()

    return fig


__all__ = [
    "show_dicom",
    "load_unspecified",
    "get_colors",
    "BGR2RGB",
    "RGB2BGR",
    "show_hist",
    "plot_average_uncertainty",
]