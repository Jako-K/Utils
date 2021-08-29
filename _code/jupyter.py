import IPython as _IPython
import matplotlib.pyplot as _plt
import torchaudio as _torchaudio

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


__all__=[
    "in_jupyter",
    "assert_in_jupyter",
    "adjust_screen_width",
    "play_audio"
]


