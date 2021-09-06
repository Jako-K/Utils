import IPython as _IPython
import matplotlib.pyplot as _plt
from PIL import Image as _Image

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
def show_image(path:str, resize_factor:float = 1.0):
    """
    Function description:
    Display a single image from `path`.

    @param path: Path to an image
    @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
    """
    # TODO Include support for URL image path

    # Checks
    _type_check.assert_types([path, resize_factor], [str, float])
    _input_output.assert_path(path)
    if resize_factor < 0:
        ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    assert_in_jupyter()
    image = _Image.open(path)

    # Resize
    width = int(image.size[0] * resize_factor)
    height = int(image.size[1] * resize_factor)
    image = image.resize((width, height), resample=0, box=None)
    display(image)



__all__=[
    "in_jupyter",
    "assert_in_jupyter",
    "adjust_screen_width",
    "play_audio"
]


