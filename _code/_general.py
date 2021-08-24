# TODO add unit tests
# TODO add intelligent search function which return possible matches and where to find them
#  e.g. search("gpu") --> ["system_info.get_vram_info()", "get_gpu_info()"]

# All around
import pandas as pd
from IPython.core.display import HTML, display
import numpy as np
from PIL import Image, ImageColor
from tkinter import Tk
from glob import glob
import shutil
import pickle
import cv2
import json
import ast

# Built-in
import datetime
from datetime import timedelta
import time
import re
import requests
import random
import platform
import pathlib
import inspect
import itertools
import math
import subprocess
import os
import sys
import calendar
import types
import warnings

# Matplotlib
import matplotlib
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn

########################################################################################################################
################################################      CHECKS       #####################################################
########################################################################################################################


class TypeChecks:

    @staticmethod
    def assert_type(to_check, expected_type, allow_none:bool=False):
        """
        Check object against expected type

        @param to_check: Object for type check
        @param expected_type: Expected type of `to_check`
        @param allow_none: Weather or not None is an accepted type or not
        """
        if not isinstance(allow_none, bool):
            raise ValueError(f"Expected `allow_None` to by of type bool, but received type `{type(allow_none)}`")

        is_ok = isinstance(to_check, expected_type)
        if allow_none:
            is_ok = (to_check is None) or is_ok

        if not is_ok:
            raise TypeError(f"Expected type `{expected_type}`, received type `{type(to_check)}`")

    # TODO Make assert list dict instead of 2 lists
    def assert_types(self, to_check:list, expected_types:list, allow_nones:list=None):
        """
        Function description:
        Check list of values against expected types

        @param to_check: List of values for type check
        @param expected_types: Expected types of `to_check`
        @param allow_nones: list of booleans or 0/1
        """

        # Checks
        self.assert_type(to_check, list)
        self.assert_type(expected_types, list)
        self.assert_type(allow_nones, list, allow_none=True)
        if len(to_check) != len(expected_types):
            ValueError("length mismatch between `to_check_values` and `expected_types`")

        # If `allow_nones` is None all values are set to False.
        if allow_nones is None:
            allow_nones = [False for _ in range(len(to_check))]
        else:
            if len(allow_nones) != len(to_check):
                ValueError("length mismatch between `to_check_values` and `nones`")
            for i, element in enumerate(allow_nones):
                if element in [0, 1]:
                    allow_nones[i] = element == 1 # the `== 1` is just to allow for zeros as False and ones as True

        # check if all elements are of the correct type
        for i, value in enumerate(to_check):
            self.assert_type(value, expected_types[i], allow_nones[i])


    def assert_list_slow(self, to_check:list, expected_type, expected_length:int=None):
        """
        Check the values of `to_check` against `expected_type` and `expected_length`.
        """
        # Tests
        self.assert_types([to_check, expected_length], [list, int])
        if expected_length < 0:
            raise ValueError(f"`expected_length >= 0, but received `{expected_length}`")
        if expected_length is not None:
            raise ValueError(f"Expected length `{expected_length}`, but received length `{len(to_check)}`")

        # check if all elements are of the correct type
        for element in to_check:
            if not isinstance(element, expected_type):
                return f"Found element of type `{type(element)}`, but expected `{expected_type}`"
type_check = TypeChecks()

########################################################################################################################
############################################      system_info       ####################################################
########################################################################################################################

class SystemInfo:
    windows_illegal_file_name_character = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]

    @staticmethod
    def get_vram_info():
        """ General information about VRAM """

        # TODO: check if ´nvidia-smi´ is installed
        # TODO: Enable multi-gpu setup i.e. cuda:0, cuda:1 ...

        # It's not necessary to understand this, it's just an extraction of info from Nvidia's API
        def get_info(command):
            assert command in ["free", "total"]
            command = f"nvidia-smi --query-gpu=memory.{command} --format=csv"
            info = output_to_list(subprocess.check_output(command.split()))[1:]
            values = [int(x.split()[0]) for i, x in enumerate(info)]
            return values[0]
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        # Format info in a readable way
        free_vram = get_info("free")
        total_vram = get_info("total")
        return {"GPU": torch.cuda.get_device_properties(0).name,
                "free": free_vram,
                "used": total_vram - free_vram,
                "total": total_vram
                }

    @staticmethod
    def get_gpu_info():
        """ Most useful things about the GPU """
        return {"name": torch.cuda.get_device_properties(0).name,
                "major": torch.cuda.get_device_properties(0).major,
                "minor": torch.cuda.get_device_properties(0).minor,
                "total_memory": torch.cuda.get_device_properties(0).total_memory / 10 ** 6,
                "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count
                }

    @staticmethod
    def get_screen_dim(WxH=True):
        """ Current screen dimensions in width X height or vice versa """

        # Get screen info through tkinter
        root = Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        return (width, height) if WxH else (height, width)

    @staticmethod
    def get_os():
        return platform.platform()

    @staticmethod
    def on_windows():
        return os.name == "nt"

    # TODO add RAM info i.e. amount and type
    def get_computer_info(self):
        """ Most useful things about the computer in general e.g. ram and gpu info """
        uname = platform.uname()

        print("=" * 40, "System Information", "=" * 40)
        print(f"System: {uname.system}")
        print(f"Node Name: {uname.node}")
        print(f"Release: {uname.release}")
        print(f"Version: {uname.version}")
        print(f"Machine: {uname.machine}")
        print(f"Processor: {uname.processor}")
        print(f"GPU: {self.get_gpu_info()['name']} - {round(self.get_gpu_info()['total_memory'])} VRAM")
        print("=" * 100)

system_info = SystemInfo()


########################################################################################################################
##########################################      Time and Date       ####################################################
########################################################################################################################


# TODO Checks
class Timer:
    """
    EXAMPLE:

    timer = Timer()

    timer.start()
    time.sleep(2)
    timer.stop()

    print(timer.get_elapsed_time())

    """


    def __init__(self, time_unit="seconds", start_on_create=False, precision_decimals=3):
        self._start_time = None
        self._elapsed_time = None
        self._is_running = False
        self._unit = None; self.set_unit(time_unit)
        self.precision_decimals = precision_decimals
        if start_on_create:
            self.start()


    def __str__(self):
        return "Timer"


    def start(self):
        if self._start_time is not None:
            self.reset()

        self._start_time = time.time()
        self._is_running = True


    def _calculate_elapsed_time(self):
        if self._start_time is None:
            return None
        else:
            return round(time.time() - self._start_time, self.precision_decimals)


    def stop(self):
        assert self._start_time is not None, "Call `start()` before `stop()`"
        self._elapsed_time = self._calculate_elapsed_time()
        self._is_running = False


    def get_elapsed_time(self):
        current_time = self._calculate_elapsed_time() if self._is_running else self._elapsed_time

        if current_time is None:
            return 0
        elif self._unit == "seconds":
            return current_time
        elif self._unit == "minutes":
            return current_time / 60.0
        elif self._unit == "hours":
            return current_time / 3600.0
        elif self._unit == "hour/min/sec":
            return str(timedelta(seconds=current_time)).split(".")[0] # the ugly bit is just to remove ms
        else:
            raise RuntimeError("Should not have gotten this far")


    def set_unit(self, time_unit:str = "hour/min/sec"):
        assert time_unit in ("hour/min/sec", "seconds", "minutes", "hours")
        self._unit = time_unit


    def reset(self):
        self._start_time = None
        self._elapsed_time = None
        self._is_running = False


# TODO Checks
class FPSTimer:
    """
    EXAMPLE:

    fps_timer = FPS_Timer()
    fps_timer.start()

    for _ in range(10):
        time.sleep(0.2)
        fps_timer.increment()
        print(fps_timer.get_fps())

    """

    def __init__(self, precision_decimals=3):
        self._start_time = None
        self._elapsed_time = None
        self.fpss = []
        self.precision_decimals = precision_decimals


    def __str__(self):
        return "FPSTimer"


    def start(self):
        assert self._start_time is None, "Call `reset()` before you call `start()` again"
        self._start_time = time.time()


    def _get_elapsed_time(self):
        return round(time.time() - self._start_time, self.precision_decimals)


    def increment(self):
        self.fpss.append(self._get_elapsed_time())


    def get_frame_count(self):
        return len(self.fpss)


    def get_fps(self, rounded=3):
        assert self._start_time is not None, "Call `start()` before you call `get_fps()`"
        if len(self.fpss) < 2:
            fps = 0
        else:
            fps = 1 / (self.fpss[-1] - self.fpss[-2])
        return round(fps, rounded)


    def reset(self):
        self._elapsed_time = None
        self.fpss = []


class TimeAndDate:
    Timer = Timer
    FPSTimer = FPSTimer

    @staticmethod
    def get_month_names(abbreviations:bool=False):
        type_check.assert_type(abbreviations, bool)

        if abbreviations:
            return [calendar.month_abbr[i].lower() for i in range(1, 13)]
        else:
            return [calendar.month_name[i].lower() for i in range(1, 13)]
time_and_date = TimeAndDate()

########################################################################################################################
##############################################       pyplot       ######################################################
########################################################################################################################

class Images:

    @staticmethod
    def is_ndarray_greyscale(image: np.ndarray):
        """ Try and determine if `image` is greyscale/has only 1 channel """
        type_check.assert_type(image, np.ndarray)
        if len(image.shape) == 2:
            return True
        elif (len(image.shape) == 3) and (len(image.shape[2]) == 1):
            return True
        else:
            return False

    @staticmethod
    def assert_ndarray_image(image: np.ndarray):
        """ Function description: Try and ensure that `image` is legitimate """
        type_check.assert_type(image, np.ndarray)
        is_greyscale = True if len(image.shape) == 2 else False

        if len(image.shape) != (2 if is_greyscale else 3):
            ValueError(f"Expected 3 dimensional shape for non-greyscale image (dim1 x dim2 x channels). "
                       f"Received shape `{image.shape}`")
        if image.min() < 0:
            ValueError("Detected a negative value in np.ndarray image, this is not allowed.")

        if image.dtype != 'uint8':
            raise TypeError(f"Expected `image` to be of dtype uint8, but received {image.dtype}. "
                            f"If `image` is in float 0-1 format instead of 0-255 8bit "
                            "try casting it with: `image = (image * 255).astype(np.uint8)`.")

        if not is_greyscale:
            num_channels = image.shape[2]
            if num_channels not in [3, 4]:
                ValueError(f"Received an unknown number of color channels: `{num_channels}`. "
                           f"Accepted number of channels are 2 and 4 (RGB and RGBA respectively) ")

    @staticmethod
    def pillow_resize_image(image: Image.Image, resize_factor: float = 1.0):
        """
        Function description:
        Resize `image` according to `resize_factor`

        @param image: Image in PIL.Image format
        @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
        @return: Resized image in PIL.Image format
        """
        type_check.assert_types([image, resize_factor], [Image.Image, float])
        width = int(image.size[0] * resize_factor)
        height = int(image.size[1] * resize_factor)
        return image.resize((width, height), resample=0, box=None)

    @staticmethod
    def pillow_image_to_ndarray(image: Image.Image, RGB2BGR: bool = True):
        """ Convert PIL image to ndarray. `RGB2BGR` is there for cv2 compatibility """
        # Tests
        type_check.assert_types([image, RGB2BGR], [Image.Image, bool])
        as_ndarray = np.asarray(image)

        return as_ndarray if not RGB2BGR else cv2.cvtColor(as_ndarray, cv2.COLOR_RGB2BGR)

    @staticmethod
    def image_size_from_path(path:str, WxH=True):
        """ Return image size in width x height or vice versa """
        io.assert_path(path)
        type_check.assert_type(WxH, bool)

        height, width = cv2.imread(path).shape[:2]
        return (width, height) if WxH else (height, width)

    @staticmethod
    def get_image_from_url(url: str, return_type: str = "cv2"):
        """
        Fetch and return image in the format specified by `return_type` from URL.
        Note that no checks are performed on the `url`
        """
        # Checks
        type_check.assert_types([url, return_type], ["str", "str"])
        if return_type not in ["pillow", "cv2"]:
            raise ValueError(f"Expected `return_type` to be in ['pillow', 'cv2'], but received `{return_type}`")

        # Download, open and return image
        image = Image.open(requests.get(url, stream=True).raw)
        return image if (return_type == "pillow") else colors.convert_color(image, "rgb")

    def ndarray_image_to_pillow(self, image: np.ndarray, BGR2RGB: bool = True):
        """ Convert ndarray image to PIL. `BGR2RGB` is there for cv2 compatibility """
        # Tests
        type_check.assert_types([image, BGR2RGB], [np.ndarray, bool])
        self.assert_ndarray_image(image)

        if BGR2RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)


    def ndarray_resize_image(self, image: np.ndarray, resize_factor: float):
        """
        Function description:
        Resize `image` according to `resize_factor`

        @param image: Image in np.ndarray format
        @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
        @return: Resized image in ndarray format
        """
        # Checks
        type_check.assert_types([image, resize_factor], [np.ndarray, float])
        self.assert_ndarray_image(image)
        if resize_factor < 0:
            ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

        new_width = int(image.shape[1] * resize_factor)
        new_height = int(image.shape[0] * resize_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


    def show_image(self, path: str, resize_factor: float = 1.0):
        """
        Function description:
        Display a single image from `path` (works in standard python and jupyter environment)

        @param path: Path to an image
        @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
        """
        # TODO Include support for URL image path

        # Checks
        type_check.assert_types([path, resize_factor], [str, float])
        io.assert_path(path)
        if resize_factor < 0:
            ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

        # If inside a jupyter environment Pillow is ued to show the image, otherwise cv2.
        if jupyter.in_jupyter():
            image = Image.open(path)
            image = self.pillow_resize_image(image, resize_factor)
            display(image)
        else:
            image = cv2.imread(path)
            image = self.ndarray_resize_image(image, resize_factor)
            self.show_ndarray_image(image)


    def show_ndarray_image(self, image: np.ndarray, resize_factor: float = 1.0, name: str = ""):
        """
        Function description:
        Display a single image from a `np.ndarray` source (works in standard python and jupyter environment)

        @param image: Image in np.ndarray format
        @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
        @param name: window name used in `cv2.imshow()` (Has no effect when inside jupyter notebook environment)
        """

        # Checks
        type_check.assert_types([image, resize_factor, name], [np.ndarray, float, str])
        self.assert_ndarray_image(image)
        if resize_factor < 0:
            ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

        # If inside a jupyter environment Pillow is ued to show the image, otherwise cv2.
        if jupyter.in_jupyter():
            image = self.ndarray_image_to_pillow(image)
            image = self.pillow_resize_image(image, resize_factor)
            display(image)
        else:
            image = self.ndarray_resize_image(image, resize_factor)
            cv2.imshow(name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def ndarray_image_center(self, image: np.ndarray, WxH: bool = True):
        """
        Function description:
        Calculate the center of an image

        @param image: Image in np.ndarray format
        @param WxH: If true width x height is returned, if false height x width
        @return: center of frame (either in WxH or HxW)
        """
        # Tests
        type_check.assert_types([image, WxH], [np.ndarray, bool])
        self.assert_ndarray_image(image)

        h, w = image.shape[:2]
        return (w // 2, h // 2) if WxH else (h // 2, w // 2)


    def cv2_cutout_square(self, image: np.ndarray, p1:tuple, p2:tuple, inverse:bool = False):
        """
        Function description:
        Cutout a square from an image or everything except a square if inverse=True

        Example:
        ##########     ##########
        ########## --> ## Black #
        ##########     ##########

        @param image: Image in np.ndarray format
        @param p1: Left upper corner of BB
        @param p2: Right lower corner of BB
        @param inverse: Determines if the background or the square is being cutout
        """
        # Checks
        type_check.assert_types([image, p1, p2, inverse], [np.ndarray, tuple, tuple, bool])
        type_check.assert_list_slow(p1, int, 2)
        type_check.assert_list_slow(p2, int, 2)
        self.assert_ndarray_image(image)

        ## Check if bb coordinates a reasonable
        h, w = image.shape[:2]
        if not ((0 < p1[0] <= w) and (0 < p2[0] <= w) and (0 < p2[1] <= h) and (0 < p2[1] <= h)):
            raise ValueError("BB coordinates `p1` and `p2` violate image width/height and/or are <= 0")

        # Make the mask (the whole `if inverse else` is just to define what is black and white)
        mask = np.ones(image.shape[:2], dtype="uint8") * (0 if inverse else 255)
        cv2.rectangle(mask, p1, p2, (255 if inverse else 0), -1)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image


    def cv2_sobel_edge_detection(self, image: np.ndarray, blur_kernel_size: int = 5,
                                 horizontal: bool = True, vertical: bool = True):
        """
        Function description:
        Perform sobel edge detection on `image` according to `blur_kernel_size`.
        Support both vertical and horizontal sobel filter together or alone.

        @param image: Image in np.ndarray format
        @param blur_kernel_size: Determine kernel size (bot height and width)
        @param horizontal: If horizontal sobel
        @param vertical: If vertical sobel
        """
        # Tests
        type_check.assert_types([image, blur_kernel_size, horizontal, vertical], [np.ndarray, int, bool, bool])
        self.assert_ndarray_image(image)
        if not (horizontal or vertical):
            raise ValueError("`horizontal` and `vertical` cannot both be False")

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

        # Sobel
        horizontal_sobel, vertical_sobel = None, None
        if horizontal:
            horizontal_sobel = cv2.Sobel(blurred, 0, 1, 0, cv2.CV_64F)
        if vertical:
            vertical_sobel = cv2.Sobel(blurred, 0, 0, 1, cv2.CV_64F)

        # Looks more mysterious then it is. Simple away of dealing with all possible values of `horizontal` and `vertical`
        edges = cv2.bitwise_or(horizontal_sobel if horizontal else vertical,
                               vertical_sobel if vertical else horizontal_sobel)
        return edges


    def cv2_draw_bounding_boxes(self, image: np.ndarray, p1: tuple, p2: tuple, label: str = None,
                                conf: float = None, color=None, line_thickness: int = 2):
        """
        EXAMPLE:
        cv2_draw_bounding_boxes(cv2_loaded_image, (438, 140), (822, 583), label="Cat", conf=0.7, color=(0,0,255))

        @param image: Image in np.ndarray format
        @param p1: Left upper corner of BB
        @param p2: Right lower corner of BB
        @param label: Description of the object surrounded by the BB e.g. "Cat"
        @param conf: BB confidence score (0-1)
        @param color: Color of the BB (must adhere to `color_helpers`'s format requirements e.g. "hex")
        @param line_thickness: Line thickness of the BB
        """

        # Checks
        type_check.assert_types(
            to_check=[image, p1, p2, label, conf, color, line_thickness],
            expected_types=[np.ndarray, tuple, tuple, str, float, object, int],
            allow_nones=[0, 0, 0, 1, 1, 1, 0, 0]
        )
        type_check.assert_list_slow(p1, int, 2)
        type_check.assert_list_slow(p2, int, 2)

        ## Check if bb coordinates a reasonable
        h, w = image.shape[:2]
        if not ((0 < p1[0] <= w) and (0 < p2[0] <= w) and (0 < p2[1] <= h) and (0 < p2[1] <= h)):
            raise ValueError("BB coordinates `p1` and `p2` violate image width/height and/or are <= 0")

        ## Check image an color
        self.assert_ndarray_image(image)
        if color is not None:
            color = colors.convert_color(color, "rgb")
        else:
            colors.random_color(max_rgb=150)  # Not to bright

        # Draw BB
        cv2.rectangle(image, p1, p2, color=color, thickness=line_thickness)

        # Draw text and confidence score
        text = ""
        if label: text += label
        if conf:
            if label: text += ": "
            text += str(round(conf * 100, 3)) + "%"
        if text:
            new_p2 = (p1[0] + 10 * len(text), p1[1] - 15)
            cv2.rectangle(image, p1, new_p2, color=color, thickness=-1)
            cv2.putText(image, text, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

########################################################################################################################
############################################       Jupyter      ########################################################
########################################################################################################################

class JupyterUtils:
    # Just convenient to have it here as well
    show_image = Images.show_image

    @staticmethod
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


    def assert_in_jupyter(self):
        if not self.in_jupyter():
            raise RuntimeError("Failed to detect Jupyter environment")


    def adjust_screen_width(self, percentage:int = 75):
        """ `percentage` determines how much of the browser's width is used by Jupyter notebook """
        # Checks
        self.assert_in_jupyter()
        type_check.assert_type(percentage, int)
        if not (50<=percentage<=100):
            raise ValueError(f"Expected 50<=`percentage`<=100, but received {percentage}")

        # Adjust width
        argument = "<style>.container { width:" + str(percentage) + "% !important; }</style>"
        display(HTML(argument))


    def jupyter_play_audio(self, path:str, plot:bool = True):
        """ Load a sound and display it if `plot` is True. Use torchaudio, so only support what they do."""
        #Tests
        self.assert_in_jupyter()
        io.assert_path(path)
        type_check.assert_type(plot, bool)

        # Audio load and play
        sound, sample_rate = torchaudio.load(path)
        audio_bar = display.Audio(path)
        display(audio_bar)

        if plot:
            duration = round(len(sound[0]) / sample_rate, 3)
            plt.plot(sound[0])
            plt.title(f"type: {audio_bar.mimetype} | duration: {duration} s | sample rate: {sample_rate}")
jupyter = JupyterUtils()

########################################################################################################################
############################################       Colors       ########################################################
########################################################################################################################

class Colors:
    """
    TODO: Fix ValueErrors which should be TypeErrors with `assert_type()`
    TODO: Add support for BGR

    Helper functions for working with colors.
    Intended use through the class instance `colors`.

    EXAMPLE:
    >> colors.random_color(color_type="hex", amount=2)
    ['#c51cbe', '#0dc76a']
    """

    # Seaborn color scheme
    seaborn_blue = (31, 119, 180)
    seaborn_orange = (255, 127, 14)
    seaborn_green = (44, 160, 44)
    seaborn_red = (214, 39, 40)
    seaborn_purple = (148, 103, 189)
    seaborn_brown = (140, 86, 75)
    seaborn_pink = (227, 119, 194)
    seaborn_grey = (127, 127, 127)
    seaborn_white = (225, 255, 255)
    seaborn_colors = {"blue": seaborn_blue,
                      "orange": seaborn_orange,
                      "green": seaborn_green,
                      "red": seaborn_red,
                      "purple": seaborn_purple,
                      "brown": seaborn_brown,
                      "pink": seaborn_pink,
                      "grey": seaborn_grey,
                      "white": seaborn_white}

    legal_types = ["rgb", "hex"]
    scheme_name_to_colors = {"seaborn": seaborn_colors}
    colors_schemes = list(scheme_name_to_colors.keys())

    @staticmethod
    def is_legal_hex(color: str):
        return isinstance(color, str) and re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color)

    @staticmethod
    def is_legal_rgb(color):
        """ Legal RGB colors are on which is a tuple/list comprised of 3 integers in the range 0-255 """
        if not (isinstance(color, list) or isinstance(color, tuple)):  # Is a list or a tuple
            return False
        if len(color) != 3:  # Has len 3
            return False
        if sum([isinstance(color_channel, int) for color_channel in color]) != 3:  # All channels is of type int
            return False
        if sum([0 <= color_channel <= 255 for color_channel in color]) != 3:  # All channels is within 0-256
            return False

        return True

    def get_color_type(self, color):
        """ Try to detect and return color type (only `legal_types` color types are supported) """
        if self.is_legal_hex(color):
            return "hex"
        elif self.is_legal_rgb(color):
            return "rgb"

        return None

    def _assert_type(self, color_type: str):
        """ assert color type is supported """
        if color_type not in self.legal_types:
            raise ValueError(f"Received unknown color type {color_type}. Legal types: {self.legal_types}")

    def assert_color(self, color):
        """ Detect color type and assert it's supported """
        if self.get_color_type(color) is None:
            raise ValueError(f"Color format cannot be interpreted")

    def _assert_color_scheme(self, scheme: str):
        """ assert color type is supported """
        if scheme not in self.colors_schemes:
            raise ValueError(f"Received unknown color scheme {scheme}. Legal types: {self.colors_schemes}")

    def _assert_color_word(self, color_name: str, scheme_name: str):
        """ Check if `color_name` is in `scheme_name` (scheme is assumed legal) """
        color_scheme = self.scheme_name_to_colors[scheme_name]
        if color_name not in color_scheme.keys():
            raise ValueError(f"Color `{color_name}` is not present in color scheme `{scheme_name}`")

    def convert_color(self, color, convert_to: str):
        """ convert color from one format to another e.g. from RGB --> HEX """
        self._assert_type(convert_to)
        self.assert_color(color)
        convert_from_type = self.get_color_type(color)

        if convert_from_type == convert_to:
            return color
        elif (convert_from_type == "rgb") and (convert_to == "hex"):
            return self.rgb_to_hex(color)
        elif (convert_from_type == "hex") and (convert_to == "rgb"):
            return self.hex_to_rgb(color)
        else:
            assert 0, "Shouldn't have gotten this far"

    def random_color(self, color_type: str = "rgb", amount: int = 1, min_rgb: int = 0, max_rgb: int = 255):
        """
        return `amount` number of random colors in accordance with `min_rgb` and `max_rgb`
        in a color format specified by `color_type`.
        """
        self._assert_type(color_type)
        if not (0 <= min_rgb <= 255):
            raise ValueError("Expected min_rgb in 0-255, received {min_rgb}")
        if not (0 <= max_rgb <= 255):
            raise ValueError("Expected max_rgb in 0-255, received {max_rgb}")
        if max_rgb <= min_rgb:
            raise ValueError("Received min_rgb > max_rgb")
        if amount < 1:
            raise ValueError("Received amount < 1")

        generated_colors = []
        for _ in range(amount):
            color = [random.randint(min_rgb, max_rgb) for _ in range(3)]
            color_converted = self.convert_color(color, color_type)
            generated_colors.append(color_converted)

        return generated_colors[0] if amount == 1 else generated_colors

    def hex_to_rgb(self, hex_color: str):
        self.assert_color(hex_color)
        return ImageColor.getcolor(hex_color, "RGB")

    def rgb_to_hex(self, rgb_color):
        self.assert_color(rgb_color)
        return "#" + '%02x%02x%02x' % tuple(rgb_color)

    def color_from_name(self, color_name: str, color_type: str = "rgb", color_scheme: str = "seaborn"):
        """
        Return color of name `color_name` from `color_scheme` in the format defined by `color_type`.
        Note: `color_name` should only contain the acutal color e.g. "blue" without prefix e.g. "seaborn_blue"
        """
        self._assert_type(color_type)
        self._assert_color_scheme(color_scheme)
        self._assert_color_word(color_name, color_scheme)

        color_scheme = self.scheme_name_to_colors[color_scheme]
        color = color_scheme[color_name]
        color_converted = self.convert_color(color, color_type)

        return color_converted

    def display_colors(self, display_colors: list):
        """ Display all colors in the list `colors in a matplotlib plot with corresponding hex, rgb etc. values"""
        if not isinstance(display_colors, list):
            raise ValueError(f"Expected type list, received type {type(display_colors)}")
        if len(display_colors) < 1:
            raise ValueError(f"Expected at least 1 color, received {len(display_colors)} number of colors")

        fig, ax = plt.subplots(figsize=(15, len(display_colors)))
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        square_height = 100 / len(display_colors)

        for i, color in enumerate(display_colors):
            self.assert_color(color)

            # matplotlib's Rectangle expect RGB channels in 0-1
            color_rgb = self.convert_color(color, "rgb")
            color_rgb_01 = [c / 255 for c in color_rgb]

            # Draw colored rectangles
            y_start = 100 - (i + 1) * square_height
            rect = matplotlib.patches.Rectangle((0, y_start), 100, square_height, color=color_rgb_01)
            ax.add_patch(rect)

            # Write colors in all legal formats
            for j, color_type in enumerate(self.legal_types):
                color_text = self.convert_color(color, color_type)
                if color_type == "rgb":
                    color_text = [" " * (3 - len(str(c))) + str(c) for c in color]
                text = f"{color_type}: {color_text}".replace("'", "")

                # White text if light color, black text if dark color + text plot
                brightness = np.mean(color_rgb)
                text_color = "black" if brightness > 50 else "white"
                plt.text(5 + j * 20, y_start + square_height // 2, text, color=text_color, size=15)

        plt.axis("off")
        plt.show()

        return fig, ax
colors = Colors()

########################################################################################################################
############################################       Imports       #######################################################
########################################################################################################################

class ImportUtils:

    @staticmethod
    def get_imports(all_requests=None):
        """
        Return common imports e.g. matplotlib.pyplot as plt.
        Expect `request` to be in `legal_imports` and to be a list e.g. ["all"]
        """
        # Checks
        if all_requests is None: all_requests = ["all"] # To avoid mutable default argument
        legal_imports = ["torch", "torchvision", "all_around", "all"]
        type_check.assert_type(all_requests, list)
        for request in all_requests:
            if request not in legal_imports:
                raise ValueError(f"Received bad request {request}. " f"Accepted requests are: {legal_imports}")

        torch_imp = \
        """
        # Torch
        import wandb
        import torch
        import torch.nn as nn
        import torch.nn.functional as F\
        """

        vision_imp = \
        """
        # Torch vision stuff
        import cv2
        import torchvision
        import albumentations as album
        import albumentations.pytorch\
        """

        all_around_imp = \
        f"""
        # All around
        import matplotlib.pyplot as plt
        import seaborn as sns; sns.set_style("darkgrid")
        {"from tqdm.notebook import tqdm" if jupyter.in_jupyter() else "from tqdm import tqdm"}
        import pandas as pd; pd.options.mode.chained_assignment = None # Avoid errors
        import numpy as np
        from glob import glob
        import sys
        import os\
        """

        # Perform import print statements
        to_concate = None
        if "all" in all_requests:
            to_concate = [torch_imp, vision_imp, all_around_imp]
        else:
            for request in all_requests:
                to_concate = []
                if request == "torchvision":
                    to_concate = [torch_imp, vision_imp]
                elif (request == "torch") and (sum([r == "torchvision" for r in all_requests]) == 0): # Avoid double torch import
                    to_concate = [torch_imp]
                elif request == "all_around":
                    to_concate = [all_around_imp]

        for all_imports in to_concate:
            [print(line[8:]) for line in all_imports.split("\n")] # <-- line[8:] remove 4 start spaces


    @staticmethod
    def get_available_functions(module):
        """ Return all public functions in `module` """
        type_check.assert_type(module, types.ModuleType)
        return [func for func, _ in inspect.getmembers(module, inspect.isfunction)]


    @staticmethod
    def get_all_available_import_classes(module):
        """ Return all public classes in `module` """
        type_check.assert_type(module, types.ModuleType)
        return [func for func, _ in inspect.getmembers(module, inspect.isclass)]


    @staticmethod
    def get_module_path(module):
        """ Return absolute path of `module` """
        type_check.assert_type(module, types.ModuleType)
        return pathlib.Path(module.__file__).resolve().parent
imports = ImportUtils()

########################################################################################################################
###########################################       Input/Output       ###################################################
########################################################################################################################

class IO:

    @staticmethod
    def assert_path(path:str):
        type_check.assert_type(path, str)
        if not os.path.exists(path):
            ValueError(f"Received bad path `{path}`")

    @staticmethod
    def assert_path_dont_exists(path:str):
        type_check.assert_type(path, str)
        if os.path.exists(path):
            ValueError(f"Path `{path}` already exists`")

    @staticmethod
    def path_exists(path:str):
        type_check.assert_type(path, str)
        return os.path.exists(path)

    @staticmethod
    def is_legal_path(path:str):
        if not isinstance(path, str):
            return False
        return os.path.exists(path)

    @staticmethod
    def add_path_to_system(path:str):
        io.assert_path(path)
        sys.path.append(path)

    @staticmethod
    def extract_file_extension(file_name:str):
        """
        Extract file extension(s) from file_name or path

        Example:
        >> extract_file_extensions("some_path/works_with_backslashes\\and_2x_extensions.tar.gz")
        '.tar.gz'
        """
        # Checks
        type_check.assert_type(file_name, str)
        if file_name.find(".") == -1:
            raise ValueError("`file_name` must contain at least 1 `.`, but received 0")

        return ''.join(pathlib.Path(file_name).suffixes)

    @staticmethod
    def get_current_directory():
        return str(pathlib.Path().absolute())

    @staticmethod
    def save_plt_plot(save_path: str, fig: matplotlib.figure.Figure = None, dpi: int = 300):
        """ Save matplotlib.pyplot figure to disk. The quality can be controlled with `dpi`"""
        # Checks
        type_check.assert_types([save_path, fig, dpi], [str, matplotlib.figure.Figure, int], [0, 1, 0])
        if (extension := io.extract_file_extension(save_path)) not in [".png", ".jpg", ".pdf"]:
            raise ValueError(f"Expected file extension to be in ['png', 'jpg', 'pdf'],"
                             f" but received `{extension}` extension")
        if fig is None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


    def get_file_basename(self, path:str, with_extension:bool = False):
        """
        Extract basename from `path`.

        Example:
        >> get_file_basename('C:/Users/JohnDoe/Desktop/test.png')
        'test.png'
        """
        # Tests
        self.assert_path(path)
        type_check.assert_type(with_extension, bool)
        if with_extension and (path.find(".") == -1):
            raise ValueError("`path` must at least contain one `.` when `with_extension` is True, but received 0")

        # Extract basename
        basename = os.path.basename(path)
        if with_extension:
            basename = basename.split(".")[-2]

        return basename


    def write_to_file(self, file_path:str, write_string:str):
        """ Append string to the end of a file """
        # Checks
        type_check.assert_type(write_string, str)
        self.assert_path(file_path)

        file = open(file_path, mode="a")
        print(write_string, file=file, end="")
        file.close()


    def read_json(self, path:str):
        """ Read .json file and return it as string"""
        self.assert_path(path)
        if path[-5:] != ".json":
            raise ValueError("Expected .json file extension, but received something else")

        f = open(path)
        data = json.load(f)
        f.close()

        return data


    def get_number_of_files(self, path:str):
        """ Return the total number of files (including folders) in `path`"""
        self.assert_path(path)
        return len(glob(os.path.join(path, "*")))


    def read_txt_file(self, path:str):
        """ Read .txt file and return it as string"""
        self.assert_path(path)
        if path[-4:] != ".txt":
            raise ValueError("Expected .txt file extension, but received something else")

        f = open(path, "r")
        fileContent = f.read()
        f.close()

        return fileContent


    def save_as_pickle(self, obj:object, file_name:str, save_path:str=None):
        """
        Save object as a pickle file in `save_path` with name `file_name`
        If `save_path` is None, the current working directory is used.
        """
        type_check.assert_types([file_name, save_path], [str, str], [0, 1])

        # Path
        if save_path is None:
            save_path = os.getcwd()
        else:
            self.assert_path(save_path)

        # Construct full name with .pkl extension
        full_path = os.path.join(save_path, file_name)
        if self.extract_file_extension(file_name).find(".pkl") == -1:
            full_path = os.path.join(full_path, ".pkl")

        with open(full_path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


    def load_pickle_file(self, path:str):
        """ Load pickle-object and return as is """
        self.assert_path(path)
        if self.extract_file_extension(path).find(".pkl") == -1:
            raise ValueError("Expected .pkl extension, but received something else")

        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)


    def copy_folder(self, from_path, to_path):
        """ Copy folder from `from_path` to `to_path`. Cannot copy folder if `to_path` already exists"""
        self.assert_path(from_path)
        self.assert_path_dont_exists(to_path)

        shutil.copytree(from_path, to_path)
        assert os.path.exists(to_path), "Something went wrong"


    # TODO: Find out if this function makes sense
    """
    def get_changed_file_name(file_path, new_file_name, new_file_extension="", path_separator="\\"):
        assert os.path.exists(file_path), "Bad path"
        assert type(new_file_extension) == str, "new_file_extension is not of type str"

        # Just a pain to deal with with backslashes
        if path_separator == "\\":
            file_path = file_path.replace("\\", "/")

        # Make all the necessary string slices
        rear_dot_index = file_path.rfind(".")
        old_extension = file_path[rear_dot_index:]
        path_before_filename_index = file_path.rfind("/")
        path_before_filename = file_path[:path_before_filename_index]
        new_path = os.path.join(path_before_filename, new_file_name)

        # Make the new name
        if (rear_dot_index == -1) and (new_file_extension == ""):
            return new_path
        elif (rear_dot_index == -1) and bool(new_file_extension):
            assert new_file_extension.find(".") != -1, "new_file_extension is missing a ´.´"
            return new_path + new_file_extension
        else:
            return new_path + old_extension
    """
io = IO()

########################################################################################################################
###########################################       Formatting      ######################################################
########################################################################################################################

class Formatting:

    @staticmethod
    def scientific_notation(number, num_mantissa:int):
        """ Rewrite number to scientific notation with `num_mantissa` amount of decimals """
        # Checks + int --> float cast if necessary
        if isinstance(number, int): number = float(number)
        type_check.assert_types([number, num_mantissa], [float, int])

        return format(number, f".{num_mantissa}E")

    @staticmethod
    def string_to_dict(string: str):
        type_check.assert_type(string, str)
        return ast.literal_eval(string)

    @staticmethod
    def string_to_list(string_list:str, element_type=None):
        """
        EXAMPLE 1:
        >> string_to_list('[198, 86, 292, 149]')
        ['198', '86', '292', '149']

        EXAMPLE 2:
        >> string_to_list('[198, 86, 292, 149]', element_type=int)
        [198, 86, 292, 149]
        """
        # Tests
        type_check.assert_types([string_list, element_type], [str, object], [0, 1])


        to_list = string_list.strip('][').split(', ')
        if element_type:
            to_list = list(map(element_type, to_list))
        return to_list
formatting = Formatting()

########################################################################################################################
###########################################       Pytorch        #######################################################
########################################################################################################################

# TODO Add function that can play audio from tensors, just like show_tensor_image()

#TODO add checks
class ArcFaceClassifier(nn.Module):
    """
    Arcface algorith:
    1. Normalize the embeddings and weights
    2. Calculate the dot products
    3. Calculate the angles with arccos
    4. Add a constant factor m to the angle corresponding to the ground truth label
    5. Turn angles back to cosines
    6. Use cross entropy on the new cosine values to calculate loss
    7. See below

    Side note: there is another 7th step in the paper that scales all the vectors by a constant
    factor at the end. This doesn't change the ordering of logits but changes their relative
    values after softmax. I didn't see it help on this dataset but it might be a
    hyperparameter worth tweaking on other tasks.
    """

    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        # Step 1: Normalize the embeddings and weights
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        # Step 2: Calculate the dot products
        return x_norm @ W_norm

#TODO add checks
def arcface_loss(cosine, target, output_classes, m=0.4):
    """
    Arcface algorith:
    1. Normalize the embeddings and weights
    2. Calculate the dot products
    3. Calculate the angles with arccos
    4. Add a constant factor m to the angle corresponding to the ground truth label
    5. Turn angles back to cosines
    6. Use cross entropy on the new cosine values to calculate loss
    7. See below

    Side note: there is another 7th step in the paper that scales all the vectors by a constant
    factor at the end. This doesn't change the ordering of logits but changes their relative
    values after softmax. I didn't see it help on this dataset but it might be a
    hyperparameter worth tweaking on other tasks.
    """

    # this prevents NaN when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
    # Step 3: Calculate the angles with arccos
    arcosine = cosine.arccos()
    # Step 4: Add a constant factor m to the angle corresponding to the ground truth label
    arcosine += F.one_hot(target, num_classes=output_classes) * m
    # Step 5: Turn angles back to cosines
    cosine2 = arcosine.cos()
    # Step 6: Use cross entropy on the new cosine values to calculate loss
    return F.cross_entropy(cosine2, target)

# TODO add checks
class RMSELoss(nn.Module):
    """ Root mean square error loss """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def seed_torch(seed:int=12, deterministic:bool=False):
    """
    Function description:
    Seed python, random, os, bumpy, torch and torch.cuda.

    @param seed: Used to seed everything
    @param deterministic: Set `torch.backends.cudnn.deterministic`. NOTE can drastically increase run time if True


    """
    type_check.assert_types([seed, deterministic], [int, bool])


    torch.backends.cudnn.deterministic = deterministic
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def show_tensor_image(image_tensor:torch.Tensor, rows:int = 1, cols:int = 1, fig_size:matplotlib.figure.Figure=(15, 10)):
    """
    Function description:
    Can show multiple tensor images.

    @param image_tensor: (samples, channels, width, height)
    @param rows: Number of rows in the plot
    @param cols: Number of columns in the plot
    @param fig_size: matplotlib.pyplot figure size (width x height)
    """
    # Tests
    type_check.assert_types([image_tensor, rows, cols, fig_size], [torch.Tensor, int, int, matplotlib.figure.Figure])
    if (rows < 1) or (cols < 1):
        raise ValueError("Both `rows` and `cols` must be greater than or equal to 1")
    if rows*cols > image_tensor.shape[0]:
        raise ValueError(f"Not enough images for {rows} rows and {cols} cols")
    if len(image_tensor.shape) != 4:
        raise ValueError(f"Expected shape (samples, channels, width, height) received {image_tensor.shape}. "
                         f"If greyscale try `image_tensor.unsqueeze(1)`")
    if image_tensor.shape[1] > 3:
        warnings.warn("Cannot handle alpha channels, they will be ignored")

    # Just to be sure
    image_tensor = image_tensor.detach().cpu()

    # If any_pixel_value > 1 --> assuming image_tensor is in [0,255] format and will normalize to [0,1]
    if sum(image_tensor > 1.0).sum() > 0:
        image_tensor = image_tensor / 255

    # Prepare for loop
    is_grayscale = True if image_tensor.shape[1] == 1 else False
    _, axs = plt.subplots(rows, cols, figsize=fig_size)
    # coordinates = [(0, 0), (0, 1), (0, 2), (1, 0) ... ]
    coordinates = list(itertools.product([i for i in range(rows)], [i for i in range(cols)]))

    for i in range(rows * cols):
        (row, col) = coordinates[i]

        # Deal with 1D or 2D plot i.e. multiple columns and/or rows
        if rows * cols == 1:
            ax = axs
        else:
            ax = axs[row, col] if (rows > 1 and cols > 1) else axs[i]

        # Format shenanigans
        image = image_tensor[i].permute(1, 2, 0).numpy()
        if np.issubdtype(image.dtype, np.float32):
            image = np.clip(image, 0, 1)

        # Actual plots
        if is_grayscale:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.axis("off")
    plt.show()


def get_device():
    # TODO: Handle multi GPU setup ?
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_parameter_count(model:nn.Module, only_trainable:bool = False, decimals:int = 3):
    """ Number of total or trainable parameters in a pytorch model i.e. nn.Module child """
    type_check.assert_types([model, only_trainable, decimals], [nn.Module, bool, int])
    if decimals < 1:
        raise ValueError(f"Expected `decimals` >= 1, but received {decimals}")

    if only_trainable:
        temp = sum(p.numel() for p in model.parameters())
    else:
        temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format(temp, f".{decimals}E")


def metric_acc(pred_logits:torch.Tensor, targets:torch.Tensor):
    """
    Function description:
    Calculate the accuracy of logits

    @param pred_logits: Prediction logits used to calculate the accuracy. Expected shape: (samples, logits)
    @param targets: Ground truth targets (ints)
    """
    type_check.assert_types([pred_logits, targets], [torch.Tensor, torch.Tensor])
    if pred_logits.shape[0] != targets.shape[0]:
        raise ValueError("Dimension mismatch between `pred_logits` and `targets`")

    preds = torch.nn.functional.softmax(pred_logits, dim=1).argmax(dim=1)
    return (preds == targets).detach().float().mean().item()


def get_batch(dataloader:torch.utils.data.DataLoader):
    """ Get the next batch """
    type_check.assert_type(dataloader, torch.utils.data.DataLoader)
    return next(iter(dataloader))


def get_model_save_name(to_add:dict, model_name:str, separator:str = "  .  ", include_time:bool = True):

    """
    Function description:
    Adds useful information to model save file such as date, time and metrics.

    Example:
    >> get_model_save_name( {"valid_loss":valid_mean}, "model.pth", "  |  ")
    "time 17.25.32 03-05-2021  |  valid_loss 0.72153  |  model_name.pth"

    @param to_add: Dictionary which contain information which will be added to the model save name e.g. loss
    @param model_name: Actual name of the model. Will be the last thing appended to the save path
    @param separator: The separator symbol used between information e.g. "thing1 <separator> thing2 <separator> ...
    @param include_time: If true, include full date and time  e.g. 17.25.32 03-05-2021 <separator> ...
    """
    # Checks
    type_check.assert_types([to_add, model_name, separator, include_time], [dict, str, str, True])
    if system_info.on_windows() and (separator in system_info.windows_illegal_file_name_character):
        raise ValueError(f"Received illegal separator symbol `{separator}`. Windows don't allow the "
                         f"following characters in a filename: {system_info.windows_illegal_file_name_character}")

    return_string = ""
    if include_time:
        time_plus_date = datetime.datetime.now().strftime('%H.%M.%S %d-%m-%Y')
        return_string = f"time {time_plus_date}{separator}" if include_time else ""

    # Adds everything from to_add dict
    for key, value in to_add.items():
        if type(value) in [float, np.float16, np.float32, np.float64]:
            value = f"{value:.5}".replace("+", "") # Rounding to 5 decimals
        return_string += str(key) + " " + str(value) + separator

    return_string += model_name
    if system_info.on_windows():
        if len(return_string) > 256:
            raise RuntimeError(f"File name is to long. Windows only allow 256 character, "
                               f"but attempted save file with `{len(return_string)} characters`")

    return return_string


# TODO refactor and add checks
def yolo_bb_from_normal_bb(bb, img_width, img_height, label, xywh=False):
    if not xywh:
        x1, y1, x2, y2 = bb
        bb_width, bb_height = (x2 - x1), (y2 - y1)
    else:
        x1, y1, bb_width, bb_height = bb

    # Width and height
    bb_width_norm = bb_width / img_width
    bb_height_norm = bb_height / img_height

    # Center
    bb_center_x_norm = (x1 + bb_width / 2) / img_width
    bb_center_y_norm = (y1 + bb_height / 2) / img_height

    # Yolo format --> |class_name center_x center_y width height|.txt  -  NOT included the two '|'
    string = str(label)
    for s in [bb_center_x_norm, bb_center_y_norm, bb_width_norm, bb_height_norm]:
        string += " " + str(s)

    return string

# TODO refactor and add checks
def yolo_draw_bbs_path(yolo_image_path, yolo_bb_path, color=(0, 0, 255)):
    assert os.path.exists(yolo_image_path), "Bad path"
    image = cv2.imread(yolo_image_path)
    dh, dw, _ = image.shape

    fl = open(yolo_bb_path, "r")
    data = fl.readlines()
    fl.close()

    for dt in data:
        _, x, y, w, h = map(float, dt.split(' '))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0: l = 0
        if r > dw - 1: r = dw - 1
        if t < 0: t = 0
        if b > dh - 1: b = dh - 1

        cv2.rectangle(image, (l, t), (r, b), color, 2)
    return image

# TODO refactor and add checks. Also merge this with `yolo_draw_bbs_path` seems wasteful to have both
def yolo_draw_single_bb_cv2(image_cv2, x, y, w, h, color=(0, 0, 255)):
    dh, dw, _ = image_cv2.shape

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0: l = 0
    if r > dw - 1: r = dw - 1
    if t < 0: t = 0
    if b > dh - 1: b = dh - 1

    cv2.rectangle(image_cv2, (l, t), (r, b), color, 2)
    return image_cv2


class _Templates:

    @staticmethod
    def _print(string):
        [print(line[8:]) for line in string.split("\n")]  # Remove the first 8 spaces from print


    def scheduler_plotter(self):
        string = \
            """
            # !git clone https://www.github.com/Jako-K/schedulerplotter
            from schedulerplotter import Plotter
            Plotter();
            """
        self._print(string)


    def common_album_aug(self):
        string = \
            """
            from albumentations.pytorch import ToTensorV2
            from albumentations import (
                HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
                IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
                IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, 
                ShiftScaleRotate, CenterCrop, Resize, MultiplicativeNoise, Solarize, MotionBlur
            )
    
            augmentations = Compose([
                RandomResizedCrop(height?, width?),
                ToTensorV2()
            ])
    
            aug_image = augmentations(image=IMAGE_NAME?)['image']
            """
        self._print(string)


    def training_loop_minimal(self):
        string = \
            """
            stats = pd.DataFrame(np.zeros((C.epochs, 2)), columns=["train_loss", "valid_loss"])
    
            epoch_progress_bar = tqdm(range(C.epochs))
            for epoch in epoch_progress_bar:
                train_losses, valid_losses = np.zeros(len(dl_train)), np.zeros(len(dl_valid))
    
                model.train()
                for i, (images, labels) in enumerate(tqdm(dl_train, leave=False)):
                    images, labels = images.to(C.device), labels.to(C.device)
    
                    # Forward pass
                    preds = model(images)
                    loss = C.criterion(preds, labels)
    
                    # Backward pass
                    loss.backward()
    
                    # Batch update and logging
                    optimizer.step()
                    optimizer.zero_grad()
                    train_losses[i] = loss.detach().cpu().item()
                    if i: epoch_progress_bar.set_postfix({"train_avg":train_losses[:i].mean()})
    
                model.eval()
                with torch.no_grad():
                    for i, (images, labels) in enumerate(tqdm(dl_valid, leave=False)):
                        images, labels = images.to(C.device), labels.to(C.device)
    
                        # Forward pass
                        preds = model(images)
                        loss = C.criterion(preds, labels)
    
                        # Batch update and logging
                        valid_losses[i] = loss.detach().cpu().item()
                        if i: epoch_progress_bar.set_postfix({"valid_avg":valid_losses[:i].mean()})
    
                # Epoch logging
                stats.iloc[epoch] = [train_losses.mean(), valid_losses.mean()]
    
            stats.plot(style="-o", figsize=(15,5))
            """

        self._print(string)


    def training_loop_with_wandb(self):
        string = \
            """
            ###################################################################################################
            #                                             Setup                                               #
            ###################################################################################################
    
            if C.use_wandb and not C.wandb_watch_activated: 
                C.wandb_watch_activated = True
                wandb.watch(model, C.criterion, log="all", log_freq=10)
    
            stats = pd.DataFrame(np.zeros((C.epochs,3)), columns=["train_loss", "valid_loss", "learning_rate"])
            best_model_name = "0_EPOCH.pth"
            print(H.get_gpu_memory_info())
    
            ###################################################################################################
            #                                          Training                                               #
            ###################################################################################################
    
            for epoch in tqdm(range(C.epochs)):
                train_losses, valid_losses = np.zeros(len(train_dl)), np.zeros(len(valid_dl))
    
                model.train()
                for i, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
                    images, labels = images.to(C.device), labels.to(C.device)
    
                    # Forward pass
                    preds = model(images)
                    loss = C.criterion(preds, labels)
    
                    # Backward pass
                    loss.backward()
    
                    # Batch update and logging
                    optimizer.step()
                    optimizer.zero_grad()
                    train_losses[i] = loss.detach().cpu().item()
    
                model.eval()
                with torch.no_grad():
                    for i, (images, labels) in enumerate(tqdm(valid_dl, leave=False)):
                        images, labels = images.to(C.device), labels.to(C.device)
    
                        # Forward pass
                        preds = model(images)
                        loss = C.criterion(preds, labels)
    
                        # Batch update and logging
                        valid_losses[i] = loss.detach().cpu().item()
    
                # Epoch update and logging
                train_mean, valid_mean, lr = train_losses.mean(), valid_losses.mean(), optimizer.param_groups[0]["lr"]
                stats.iloc[epoch] = [train_mean, valid_mean, lr]
                scheduler.step()
                C.epochs_trained += 1
    
                if C.use_wandb:
                    wandb.log({"train_loss": train_mean, "valid_loss": valid_mean, "lr":lr})
    
                if (epoch > 0) and (stats["valid_loss"][epoch] < stats["valid_loss"][epoch-1]): # Save model if it's better
                    extra_info = {"valid_loss":valid_mean, "epochs_trained":C.epochs_trained}
                    best_model_name = T.get_model_save_name(extra_info, "model.pth", include_time=True)
                    torch.save(model.state_dict(), best_model_name)
    
            ###################################################################################################
            #                                          Finish up                                              #
            ###################################################################################################
    
            # Plot
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
            stats.drop(columns=["learning_rate"]).plot(ax=ax1, style="-o")
            ax2.ticklabel_format(style="scientific", axis='y', scilimits=(0,0))
            stats["learning_rate"].plot(ax=ax2, style="-o")
    
            # Save model
            if C.use_wandb:
                import shutil
                shutil.copy(best_model_name, os.path.join(wandb.run.dir, best_model_name))
                wandb.save(best_model_name, policy="now")
                    """
        self._print(string)


    def wandb(self):
        string = \
            """
            # 1.) Config file   -->  plain and simple `dict` can be used
            # 2.) wandb.init()  -->  Starts project and spin up wandb in the background
            # 3.) wandb.watch() -->  Track the torch model's parameters and gradients over time
            # 4.) wandb.log()   -->  Logs everything else like loss, images, sounds ... (super versatile, even takes 3D objects) 
            # 5.) wandb.save()  -->  Saves the model
            """
        self._print(string)


    def config_file(self):
        string = \
            """
            class Config:
                # Control
                mode = "train"
                debug = False
                use_wandb = False
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
                # Path
                main_path = ""; assert os.path.exists(main_path)
                csv_path = ".csv"; assert os.path.exists(csv_path)
                model_load_path = ".pth"; assert os.path.exists(model_load_path)
    
                # Adjustable variables
                wandb_watch_activated = False
                epochs_trained = 0
    
                # Hypers
                batch_size = 32
                epochs = 10
                criterion = nn.CrossEntropyLoss()
                optimizer_hyper = dict(lr = 5e-4)
                optimizer = torch.optim.Adam
                scheduler_hyper = dict(lr_lambda = lambda epoch: 0.85 ** epoch)
                scheduler = torch.optim.lr_scheduler.LambdaLR
    
                # Seed everything
                seed = 12
                T.seed_torch(seed)
    
                to_log = dict(
                    seed = seed,
                    mode = mode,
                    debug = debug,
                    device = device,
                    epochs=epochs,
                    batch_size = batch_size,
                    criterion = criterion,
                    optimizer = (optimizer, optimizer_hyper),
                    scheduler = (scheduler, scheduler_hyper),
                    dataset="MNIST",
                    architecture="Resnet18",
                    notes="resnet 18 is pretrained ... expect to see ... using ABC is questionable"
                )
    
            C = Config()
            if C.use_wandb:
                wandb.login()
                wandb.init(project=?, config=C.to_log)
    
            """
        self._print(string)


class Pytorch:
    templates = _Templates()

    # nn.Module
    ArcFaceClassifier = ArcFaceClassifier
    arcface_loss = arcface_loss
    RMSELoss = RMSELoss

    # Utils
    seed_torch = seed_torch
    show_tensor_image = show_tensor_image
    get_device = get_device
    get_parameter_count = get_parameter_count
    metric_acc = metric_acc
    get_batch = get_batch
    get_model_save_name = get_model_save_name

    # Yolo
    yolo_bb_from_normal_bb = yolo_bb_from_normal_bb
    yolo_draw_bbs_path = yolo_draw_bbs_path
    yolo_draw_single_bb_cv2 = yolo_draw_single_bb_cv2
pytorch = Pytorch()

########################################################################################################################
###########################################       All around       #####################################################
########################################################################################################################

class AllAround:

    @staticmethod
    def sturges_rule(data):
        """
        Function description:
        Uses Sturges' rule to calculate bin_width and number_of_bins

        @param data: To be binned. `data` must have well defined: len, min and max

        Example:
        >> AllAround.sturges_rule([1,2,3,4])
        (1.004420127756955, 3)
        """
        # NOTE Not sure about the intuition for the method or even how well it works, but
        # it seems like a reasonable way of picking bin sizes (and therefore #bins)

        try:
            len(data), min(data), max(data)
        except TypeError:
            raise TypeError("`data` must have well defined len, min and max")

        k = 1 + 3.3 * np.log10(len(data))
        bin_width = (max(data) - min(data)) / k
        number_of_bins = int(np.round(1 + np.log2(len(data))))

        return bin_width, number_of_bins

    @staticmethod
    def pandas_standardize_df(df:pd.DataFrame):
        """
        Standardize pandas DataFrame

        Example:
        >> all_around.pandas_standardize_df(pd.DataFrame(np.array([1,2,3,4])))
                  0
        0 -1.161895
        1 -0.387298
        2  0.387298
        3  1.161895
        """
        type_check.assert_type(df, pd.DataFrame)
        df_standardized = (df - df.mean()) / df.std()
        assert np.isclose(df_standardized.mean(), 0), "Expected mean(std) ~= 0"
        assert np.isclose(df_standardized.std(), 1), "Expected std(std) ~= 1"
        return df_standardized

    @staticmethod
    def get_grid_coordinates(rows: int, cols: int):
        """
        Calculate 2D coordinates for grid traversing. If unclear, the example below should alleviate any doubt

        Example:

        >> get_grid_coordinates(3,2)
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        """
        type_check.assert_types([rows, cols], [int, int])
        if (rows < 1) or (cols < 1):
            raise ValueError(
                f"Expected both rows and cols to be positive, but received cols: `{cols}` and rows: `{rows}`")
        return list(itertools.product([i for i in range(rows)], [i for i in range(cols)]))

    @staticmethod
    def unfair_coin_flip(p: float):
        """
        Function description:
        Flip a weighted coin

        @param p: Percentage of success should be range (0, 1)
        """
        type_check.assert_type(p, float)
        if not (0.0<p<1.0):
            raise ValueError(f"0<p<1, but received p: `{p}`")
        return random.random() > p

    @staticmethod
    def int_sign(x: int):
        """ Sign (i.e. + or -) of an integer """
        type_check.assert_type(x, int)
        return math.copysign(1, x)

    @staticmethod
    def init_2d_array(rows, cols, value=None):
        """
        EXAMPLE:

        >> all_around.init_2d_array(4,3)
        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        """
        return [[value for _ in range(cols)] for _ in range(rows)]

    def ndarray_to_bins(self, array:np.ndarray, num_bins:int = None):
        """
        Function description:
        Bin `array` into `num_bins` number of bins.

        @param array: Numpy array containing the values which is going to be binned
        @param num_bins: The number of bins used. If bins is None Sturges rule will be used automatically
        @return: array_binned (`array` binned), num_bins (the number of bins), thresholds (Thresholds used to bin)

        Example:
        >> all_around.ndarray_to_bins(np.array([1,2,3,4]), 2)
        (array([1, 1, 2, 3], dtype=int64), 2, array([1. , 2.5, 4. ]))
        """
        type_check.assert_types([array, num_bins], [np.ndarray, int], [0, 1])

        if num_bins is None:
            num_bins = self.sturges_rule(array)[1]

        _, thresholds = np.histogram(array, bins=num_bins)
        array_binned = np.digitize(array, thresholds)
        return array_binned, num_bins, thresholds

    # TODO refactor, checks and remove underscore
    @staticmethod
    def _plot_average_uncertainty(data, stds=2):
        """
        data: np.array with shape (samples X repetitions)
        """
        xs = np.arange(len(data))
        std = data.std(1)
        mean = data.mean(1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("Individual")
        ax1.plot(data, ".-")

        ax2.set_title("Averaged with uncertainty")
        ax2.plot(mean, 'o-', color=colors.color_from_name("blue", "hex"), label='Mean')
        plt.sca(ax2)  # <-- makes gca work, super weird but gets the job done
        plt.gca().fill_between(xs, mean - stds * std, mean + stds * std, color='lightblue', alpha=0.5,
                               label=r"$2\sigma$")
        plt.plot(xs, [mean.mean()] * len(xs), '--', color=colors.color_from_name("orange", "hex"),
                 label="Mean of means")
        ax2.legend()
        plt.show()

        return fig
all_around = AllAround()


__all__ = [
    "type_check",
    "system_info",
    "time_and_date",
    "Images",
    "jupyter",
    "colors",
    "imports",
    "io",
    "formatting",
    "pytorch",
    "all_around",
]


if __name__ == "__main__":
    pass
    #all_around.ndarray_to_bins(np.array([1, 2, 3, 4]))