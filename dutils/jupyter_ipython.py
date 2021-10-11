import IPython as _IPython
import cv2 as _cv2
import matplotlib.pyplot as _plt
from PIL import Image as _Image
import numpy as _np
import requests as _requests
import validators as _validators

from . import type_check as _type_check
from . import input_output as _input_output


def in_jupyter():
    """
    Check if python is currently running in a jupyter enviroment.
    NOTE: The implementation is somewhat dubious, but have tested it to the best of my abilities
    """

    try:
        shell = get_ipython().__class__.__name__ # This is supposed to be an unresolved reference anywhere outside jupyter
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?), going to assume that we are not in jupyter
    except NameError:
        return False  # Probably standard Python interpreter


def assert_in_jupyter():
    if not in_jupyter():
        raise RuntimeError("Failed to detect Jupyter environment")


def adjust_screen_width(percentage:int=75):
    """
    Adjust the cell width of Jupyter notebook. 100 % is the entire browser window.

    @param percentage: Percentage cells are scaled by (min: 50, max: 100)
    @return: None
    """
    # Checks
    _type_check.assert_type(percentage, int)
    assert_in_jupyter()
    if not (50<=percentage<=100):
        raise ValueError(f"Expected 50 <= `percentage` <= 100, but received `{percentage}`")

    # Adjust width
    argument = "<style>.container { width:" + str(percentage) + "% !important; }</style>"
    _IPython.core.display.display(_IPython.core.display.HTML(argument))


def play_audio(path:str, plot:bool=True):
    """
    Load sound from `path` and display it if `plot` is True.
    NOTE: Uses torchaudio as backend, and therefore only supports what they do.

    @param path: Path where sound can be found
    @param plot: Determine if the raw waveform is plotted alongside the playable audio
    @return: None
    """

    import torchaudio as _torchaudio # TODO Move this together with the other imports, when the "sox warning shenanigans" has been fixed

    #Tests
    assert_in_jupyter()
    _input_output.assert_path(path)
    _type_check.assert_type(plot, bool)

    # Audio load and play
    sound, sample_rate = _torchaudio.load(path)
    audio_bar = _IPython.display.Audio(path)
    _IPython.core.display.display(audio_bar)

    if plot:
        duration = round(len(sound[0]) / sample_rate, 3)
        _plt.plot(sound[0])
        _plt.title(f"type: {audio_bar.mimetype} | duration: {duration} s | sample rate: {sample_rate}")


# TODO add unit test
def show_image(source, resize_factor:float=1.0, BGR2RGB:bool=None):
    """
    Display a single image from path, np.ndarray or url.

    @param source: path, np.ndarray or url pointing to the image you wish to display
    @param resize_factor: Rescale factor in percentage (i.e. 0-1), `scale_factor` < 0
    @param BGR2RGB: Convert `source` from BGR to RGB. If `None`, will convert np.ndarray images automatically
    """

    # Simple checks
    _type_check.assert_in(type(source), [_np.ndarray, str])
    _type_check.assert_types([resize_factor, BGR2RGB], [float, bool], [0, 1])
    assert_in_jupyter()

    # `source` and `resize` checks
    is_path = _input_output.path_exists(source) if isinstance(source, str) else False
    is_url = True if isinstance(source, str) and _validators.url(source) is True else False
    is_ndarray = True if isinstance(source, _np.ndarray) else False

    if not (is_path or is_url or is_ndarray):
        raise ValueError("`source` could not be interpreted as a path, url or ndarray.")
    if is_path + is_url + is_ndarray > 1:
        raise AssertionError("This should not be possible") # Don't see how a path and a url can be valid simultaneously
    if resize_factor < 0:
        raise ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    if is_path:
        image = _Image.open(source)
    elif is_url:
        image = _Image.open(_requests.get(source, stream=True).raw)
    elif is_ndarray:
        image = _Image.fromarray(source)

    # Swap blue and red color channel stuff
    num_channels = len(image.getbands())
    bgr2rgb_auto = (BGR2RGB is None) and is_ndarray and (num_channels in [3, 4])
    if BGR2RGB or bgr2rgb_auto:
        # BGR --> RGB or BGRA --> RGBA
        as_array = _np.asarray(image)
        color_corrected = _cv2.cvtColor(as_array, _cv2.COLOR_BGR2RGB) if (num_channels == 3) \
                          else _cv2.cvtColor(as_array, _cv2.COLOR_BGRA2RGBA)
        image = _Image.fromarray(color_corrected)

    if resize_factor != 1.0:
        width = int(image.size[0] * resize_factor)
        height = int(image.size[1] * resize_factor)
        image = image.resize((width, height), resample=0, box=None)

    display(image)


__all__=[
    "in_jupyter",
    "assert_in_jupyter",
    "adjust_screen_width",
    "play_audio",
    "show_image"
]


