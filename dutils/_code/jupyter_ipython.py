import IPython as _IPython
import matplotlib.pyplot as _plt
from PIL import Image as _Image
import numpy as _np
import requests as _requests
import validators as _validators

from . import type_check as _type_check
from . import input_output as _input_output


# Just convenient to have it here as well
# show_image = Images.show_image

def in_jupyter():
    # Not the cleanest, but gets the job done
    try:
        shell = get_ipython().__class__.__name__ # This is supposed to be an unresolved reference
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def assert_in_jupyter():
    if not in_jupyter():
        raise RuntimeError("Failed to detect Jupyter environment")


def adjust_screen_width(percentage:int=75):
    """ `percentage` determines how much of the browser's width is used by Jupyter notebook """

    # Checks
    _type_check.assert_type(percentage, int)
    assert_in_jupyter()
    if not (50<=percentage<=100):
        raise ValueError(f"Expected 50 <= `percentage` <= 100, but received `{percentage}`")

    # Adjust width
    argument = "<style>.container { width:" + str(percentage) + "% !important; }</style>"
    _IPython.core.display.display(_IPython.core.display.HTML(argument))


def play_audio(path:str, plot:bool=True):
    """ Load a sound and display it if `plot` is True. Use torchaudio, so only support what they do."""
    import torchaudio as _torchaudio # Move this to other imports, when the "sox warning shenanigans" has been fixed

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
    Function description:
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
        raise ValueError("`source` could not be intepreted as a path, url or ndarray.")
    if is_path + is_url + is_ndarray > 1:
        raise AssertionError("This should not be possible") # Don't see how a path and an url can be valid simultaneously
    if resize_factor < 0:
        ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    if is_path:
        image = _Image.open(source)
    elif is_url:
        image = _Image.open(_requests.get(source, stream=True).raw)
    elif is_ndarray:
        image = _Image.fromarray(source)

    bgr2rgb_auto = BGR2RGB is None and is_ndarray
    if BGR2RGB or bgr2rgb_auto:
        # BGR --> RGB (done in numpy, just because it's the easiest)
        image = _Image.fromarray(_np.asarray(image)[:, :, ::-1])

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


