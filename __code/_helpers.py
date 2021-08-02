import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torchaudio
warnings.filterwarnings("default", category=UserWarning)

import torch
import IPython
import random
import os
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import pathlib
import shutil
import pickle
import numpy as np
import cv2
import inspect
import pathlib
import psutil
import platform
from tkinter import Tk
import itertools
import json
import ast
import requests
import math
import subprocess
import os

def in_jupyter():
    # Not the cleanest, but gets the job done
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def on_windows():
    return os.name == "nt"


def sturges_rule(data):
    """
    Data must have well defined: len, min and max
    Not sure about the intuition for the method or even how well it works, but
    it seems like a reasonble way of picking bins sizes (and therefore #bins) 
    """
    k = 1 + 3.3 * np.log10(len(data))
    optimal_class_width = ( max(data) - min(data) ) / k
    number_of_bins = int(np.round(1 + np.log2(len(data))))
    
    return optimal_class_width, number_of_bins


def get_gpu_memory_info():
    """ Return the systems total amount of VRAM along with current used/free VRAM"""
    # TODO: check if ´nvidia-smi´ is installed 
    # TODO: Enable multi-gpu setup i.e. cuda:0, cuda:1 ...
    
    def get_info(command):
        assert command in ["free", "total"]
        command = f"nvidia-smi --query-gpu=memory.{command} --format=csv"
        info = output_to_list(subprocess.check_output(command.split()))[1:]
        values = [int(x.split()[0]) for i, x in enumerate(info)]
        return values[0]
        
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    free_vram = get_info("free")
    total_vram = get_info("total")
    return {"GPU":torch.cuda.get_device_properties(0).name,
            "free": free_vram, 
            "used": total_vram-free_vram, 
            "total":total_vram
            }
    
def get_gpu_info():
    return {"name" : torch.cuda.get_device_properties(0).name,
            "major" : torch.cuda.get_device_properties(0).major,
            "minor" : torch.cuda.get_device_properties(0).minor,
            "total_memory" : torch.cuda.get_device_properties(0).total_memory/10**6,
            "multi_processor_count" : torch.cuda.get_device_properties(0).multi_processor_count
            }


def get_general_computer_info():
    uname = platform.uname()

    print("="*40, "System Information", "="*40)
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    print(f"GPU: {get_gpu_info()['name']} - {round(get_gpu_info()['total_memory'])} VRAM")
    print("="*100)


def expand_jupyter_screen(percentage:int = 75):
    assert in_jupyter(), "Most be in Jupyer notebook"
    assert percentage in [i for i in range(50,101)], "Bad argument" # Below 50 just seems odd, assumed to be a mistake
    from IPython.core.display import display, HTML
    argument = "<style>.container { width:" + str(percentage) + "% !important; }</style>"
    display(HTML(argument))


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


class FPS_Timer:
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
        return round(fps, 3)


    def reset(self):
        self._elapsed_time = None
        self.fpss = []


def show_image(path:str, resize_factor:float=1.0):
    assert os.path.exists(path), "Bad path"
    assert resize_factor > 0, "resize_factor must have a value greater than 0"

    if in_jupyter():
        img = Image.open(path)
        img = pillow_resize_image(img, resize_factor)
        display(img)
    else:
        img = cv2.imread(path)
        img = cv2_resize_image(img, resize_factor)
        cv2_show_image(img)


def play_audio(path:str, plot:bool = True):
    assert os.path.exists(path), "Bad path"
    assert in_jupyter(), "Most be in Jupyer notebook"
    sound, sample_rate = torchaudio.load(path)
    
    # Audio
    audio_bar = IPython.display.Audio(path)
    display(audio_bar)
    
    # Plot
    if plot:
        duration = round(len(sound[0])/sample_rate,3)
        plt.plot(sound[0])
        plt.title(f"type: {audio_bar.mimetype} | duration: {duration} s | sample rate: {sample_rate}")


def read_txt_file(path):
    assert os.path.exists(path), "Bad path"
    assert path[-4:] == ".txt", "Wrong file format, expected ´.txt´"
    objFile = open(path, "r")
    fileContent = objFile.read();
    objFile.close()
    
    return fileContent


def save_as_pickle_file(obj:object, file_name:str, save_path:str=None):
    assert obj is not None and file_name is not None, "Bad argument(s)"
    if save_path is None:
        save_path = os.getcwd()
    else:
        assert os.path.exists(save_path), "Bad path"
    full_path = os.path.join(save_path, file_name)

    with open(f'{full_path}.pkl', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pickle_file(path: str):
    assert os.path.exists(path), "Bad path"
    with open(path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

class _ColorRGB:
    blue = (31, 119, 180)
    orange = (255, 127, 14)
    green = (44, 160, 44)
    red = (214, 39, 40)
    purple = (148, 103, 189)
    brown = (140, 86, 75)
    pink = (227, 119, 194)
    grey = (127, 127, 127)
    white = (225, 255, 255)
    all_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'white']

    def random_color(self, only_predefined_colors=True):
        if only_predefined_colors:
            return getattr(self, random.choice(self.all_colors))
        else:
            return [random.randint(0, 255) for i in range(3)]

    def random_not_to_bright_color(self):
        return [random.randint(0, 150) for i in range(3)]


    def is_legal(self, color):
        for color_channel in color:
            if not(0 <= color_channel <= 255):
                return False
        return True
colors_rgb = _ColorRGB()


class _ColorHEX:
    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    red = "#d62728"
    purple = "#9467bd" 
    brown = "#8c564b"
    pink = "#e377c2"
    grey =  "#7f7f7f"
colors_hex = _ColorHEX()

def copy_folder(from_path, to_path):
    assert not os.path.exists(to_path), "shutil don't allow ´to_path´ to already exist"
    shutil.copytree(from_path, to_path)
    assert os.path.exists(to_path), "Something went wrong"


def get_imports(request="all"):
    legal = ["torch", "torchvision", "all_around", "path", "file", "all"]
    if isinstance(request, str):
        assert request in legal
        requests = [request]
    elif isinstance(request, list):
        include_all = False
        for single_request in [r.lower() for r in request]:
            assert single_request in legal, "bad request"
            if single_request == "all":
            	include_all = True
        requests = request if not include_all else ["all"]
    
    torch_imp = \
    """
    # Torch
    import wandb
    import torch
    import torch.nn as nn
    import utilsm.torch_helpers as T
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
    """
    # All around
    import matplotlib.pyplot as plt
    import seaborn; seaborn.set_style("darkgrid")
    from tqdm.notebook import tqdm
    import pandas as pd
    import numpy as np
    import os
    import wandb\
    """

    path_imp = \
    """
    # Path and file stuff
    from pathlib import Path
    import pickle
    import glob
    import sys
    import os\
    """
    
    for request in requests:
        to_concate = []
        if request == "all": 
        	to_concate = [torch_imp, vision_imp, all_around_imp, path_imp]
        elif request == "torchvision": 
        	to_concate = [torch_imp, vision_imp]
        elif (request == "torch") and (sum([r == "torchvision" for r in requests]) == 0): # Avoid double torch import 
        	to_concate = [torch_imp]
        elif request == "all_around": 
        	to_concate = [all_around_imp]   
        elif request in ["path", "file"]: 
        	to_concate = [path_imp]   

        return_string = ""
        for imports in to_concate:
            [print(line[4:]) for line in imports.split("\n")] # <-- line[4:] remove 4 start spaces


def extract_file_extension(file_name:str):
	"""
	>>> extract_file_extensions("some_path/works_with_backslashes\\and_2x_extensions.tar.gz")
	'.tar.gz'
	"""
	assert file_name.find(".") != -1, "No ´.´ found"
	suffixes = pathlib.Path(file_name).suffixes
	return ''.join(pathlib.Path(file_name).suffixes)


def plot_average_uncertainty(data, stds=2):
    """
    data: np.array with shape (samples X repetitions)
    """
    xs = np.arange(len(data))
    std = data.std(1)
    mean = data.mean(1)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
    ax1.set_title("Individual")
    ax1.plot(data, ".-")
    
    ax2.set_title("Averaged with uncertainty")
    ax2.plot(mean, 'o-', color=colors_hex.blue, label='Mean')
    plt.sca(ax2) # <-- makes gca work, super wierd but gets the job done
    plt.gca().fill_between(xs, mean - stds*std, mean + stds*std,  color='lightblue', alpha=0.5, label=r"$2\sigma$")
    plt.plot(xs, [mean.mean()]*len(xs), '--', color=colors_hex.orange, label="Mean of means")
    ax2.legend()
    plt.show()

    return fig


def to_bins(target, bins="auto"):
    if bins == "auto":
        _, bins = sturges_rule(target)
    
    count, division = np.histogram(target, bins=bins)
    groups = np.digitize(target, division)
    return groups, bins, division


def write_to_file(file_path:str, write_string:str, only_txt:bool = True):
    """ Appends a string to the end of a file"""
    if only_txt:
        assert extract_file_extension(file_path) == ".txt", "´only_txt´ = true, but file type is not .txt"
    
    file = open(file_path, mode="a")
    print(write_string, file=file, end="")
    file.close()


def scientific_notation(number, num_mantissa=4):
    assert f"{number}".isnumeric() and f"{num_mantissa}".isnumeric() 
    return format(number, f".{num_mantissa}E")


def save_plt_plot(save_path, fig=None, dpi=300):
    assert extract_file_extension(save_path) in [".png", ".jpg", ".pdf"]
    if fig is None:
        plt.savefig(save_path, dpi = dpi, bbox_inches = 'tight')
    else:
        fig.savefig(save_path, dpi = dpi, bbox_inches = 'tight')


def plot_lambda(lambda_f, a:int=None, b:int=None):
    """ TODO: find a better solution than a,b = -10,10 when None"""
    if a is None: a = -10
    if b is None: b = 10
    xs = np.linspace(a, b, 500)
    ys = [lambda_f(x) for x in xs]
    plt.plot(xs, ys)


def cv2_show_image(image, resize_factor=1.0, name=""):
    assert resize_factor > 0, "resize_factor must have a value greater than 0"

    if in_jupyter():
        img = cv2_image_to_pillow(image)
        img = pillow_resize_image(img, resize_factor)
        display(img)
    else:
        img = cv2_resize_image(image, resize_factor)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def unfair_coin_flip(p):
    return random.random() > p


def get_all_available_import_functions(module):
    return [func for func, _ in inspect.getmembers(module, inspect.isfunction)]


def get_all_available_import_classes(module):
    return [func for func, _ in inspect.getmembers(module, inspect.isclass)]


def get_current_working_directory():
    return str(pathlib.Path().absolute())


def get_changed_file_name(file_path, new_file_name, new_file_extension="", path_separator="\\"):
    assert os.path.exists(file_path), "Bad path"
    assert type(new_file_extension) == str, "new_file_extension is not of type str"

    # Just a pain to deal with with backslashes
    if path_separator == "\\":
        file_path = file_path.replace("\\", "/")

    # Make all the necesarry string slices
    rear_dot_index = file_path.rfind(".")
    old_extension = file_path[rear_dot_index:]
    path_before_filename_index = file_path.rfind("/")
    path_before_filename = file_path[:path_before_filename_index]
    new_path = os.path.join( path_before_filename, new_file_name)
    
    # Make the new name
    if (rear_dot_index == -1) and (new_file_extension == ""):
        return new_path
    elif (rear_dot_index == -1) and bool(new_file_extension):
        assert new_file_extension.find(".") != -1, "new_file_extension is missing a ´.´"
        return new_path + new_file_extension
    else:
        return new_path + old_extension


def number_of_files_in(folder_path):
    return len(glob.glob( os.path.join(folder_path, "*")))


def get_module_path(module):
    return pathlib.Path(module.__file__).resolve().parent


def cv2_resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)


def pillow_resize_image(image, scale_factor):
    width = int(image.size[0] * scale_factor)
    height = int(image.size[1] * scale_factor)
    return image.resize((width, height), resample=0, box=None)


def get_current_screen_dimensions(WxH=True):
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    if WxH:
        return width, height
    else:
        return height, width


def cv2_sobel_edge_detection(img, blur_kernel=(5, 5)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    horizontal = cv2.Sobel(blurred, 0, 1, 0, cv2.CV_64F)
    vertical = cv2.Sobel(blurred, 0, 0, 1, cv2.CV_64F)
    edges = cv2.bitwise_or(horizontal, vertical)
    return edges


def init_2d_array(rows, cols, value=None):
    """
    EXAMPLE:
    
    >>> init_2d_array(4,3)
    [[None, None, None],
     [None, None, None],
     [None, None, None],
     [None, None, None]]
    """
    return [ [value for _ in range(cols)] for _ in range(rows)]


def get_grid_coordinates(rows, cols):
    """
    EXAMPLE:

    >>> get_grid_coordinates(3,2)
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    """
    return list(itertools.product([i for i in range(rows)], [i for i in range(cols)]))


def cv2_frame_center(frame, WxH=True):
    h, w = frame.shape[:2]
    return  (w//2, h//2) if WxH else (h//2, w//2)


def cv2_remove_everything_except_square(image, x1, y1, x2, y2, color=255):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
    masked = cv2.bitwise_and(image, image, mask=mask) #Apply mask
    return masked


def read_json(path):
    assert os.path.exists(path), "Bad path"
    assert (path[-5:] == ".json"), "Bad extension, expected .json"

    f = open(path)
    data = json.load(f)
    f.close()
    
    return data


def string_to_dict(string:str):
    return ast.literal_eval(string)


def cv2_image_to_pillow(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        return Image.fromarray(img)
    except TypeError:
        # Wrong format. Gets 0-1 expects 0-255 8bit int.
        im = Image.fromarray((img * 255).astype(np.uint8)) 

def pillow_image_to_cv2(image, RGB2BGR=True):
    img_cv2 = np.asarray(image)
    return img_cv2 if not RGB2BGR else cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)


def cv2_draw_bounding_boxes(image, p1, p2, label=None, conf=None, color="random", line_thickness=2, text_color=(200,200,200)):
    """
    EXAMPLE:
    cv2_draw_bounding_boxes(cv2_loaded_image, (438, 140), (822, 583), label="Cat", conf=0.7, color=(0,0,255))
    """
    
    if color == "random": 
        color = colors_rgb.random_not_to_bright_color()
    elif color[0] == "#":
        color = hex_color_to_rgb(color)
    
    cv2.rectangle(image, p1, p2, color=color, thickness=line_thickness)
    
    text = ""
    if label:
        text += label
    if conf:
        if label:
            text += ": "
        text += str( round(conf*100, 3) ) + "%"
    
    if label or conf:
        new_p2 = (p1[0]+10*len(text), p1[1]-15)
        cv2.rectangle(image, p1, new_p2, color=color, thickness=-1)
        cv2.putText(image, text, (p1[0], p1[1]-2), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def hex_color_to_rgb(hex_color):
    return ImageColor.getcolor(hex_color, "RGB")


def string_to_list(string_list, element_type=None):
    """
    EXAMPLE 1:
    ['198', '86', '292', '149'] = string_to_list('[198, 86, 292, 149]')
    
    EXAMPLE 2:
    [198, 86, 292, 149] = string_to_list('[198, 86, 292, 149]', element_type=int)
    """
    
    to_list = string_list.strip('][').split(', ')
    if element_type:
        to_list = list(map(element_type, to_list))
    return to_list



def get_image_size(path, WxH=True):
    assert os.path.exists(path), "Bad path"
    height, width = cv2.imread(path).shape[:2]
    return (width, height) if WxH else (height, width)


def normal_bb_coordinates_to_yolo_format(bb, img_width, img_height, label, xywh=False):
    if not xywh:
        x1, y1, x2, y2 = bb
        bb_width, bb_height = (x2-x1), (y2-y1)
    else:
        x1, y1, bb_width, bb_height = bb
        
    # Width and height
    bb_width_norm = bb_width/img_width
    bb_height_norm = bb_height/img_height
    
    # Center
    bb_center_x_norm =  (x1 + bb_width/2) / img_width
    bb_center_y_norm =  (y1 + bb_height/2) / img_height
    
    # Yolo format --> |class_name center_x center_y width height|.txt  -  NOT included the two '|'
    string = str(label)
    for s in [bb_center_x_norm, bb_center_y_norm, bb_width_norm, bb_height_norm]:
        string += " " + str(s)
        
    return string


def yolo_draw_bbs_path(yolo_image_path, yolo_bb_path, color = (0,0,255)):
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


def yolo_draw_single_bb_cv2(image_cv2, x, y, w, h, color=(0,0,255)):
    dh, dw, _ = image_cv2.shape

    l = int( (x - w/2) * dw)
    r = int( (x + w/2) * dw)
    t = int( (y - h/2) * dh)
    b = int( (y + h/2) * dh)

    if l < 0: l = 0
    if r > dw - 1: r = dw - 1
    if t < 0: t = 0
    if b > dh - 1: b = dh - 1

    cv2.rectangle(image_cv2, (l, t), (r, b), color, 2)
    return image_cv2


def get_image_from_url(url:str, return_type="cv2"):
    assert return_type in ["pillow", "cv2"], "`return_type` not in ['pillow', 'cv2']"
    if return_type == "cv2":
        return np.asarray(Image.open(requests.get(url, stream=True).raw))
    elif return_type == "pillow":
        return Image.open(requests.get(url, stream=True).raw)



def int_sign(x:int):
    return math.copysign(1, x)


def pandas_standardize_df(df):
    df_standardized = (df - df.mean()) / df.std()
    assert np.isclose(df_standardized.mean(), 0), "mean(std) ~= 0"
    assert np.isclose(df_standardized.std(), 1), "std(std) ~= 1"
    return df_standardized




# Check __all__ have all function ones in a while
# [func for func, _ in inspect.getmembers(H, inspect.isfunction)]
# [func for func, _ in inspect.getmembers(H, inspect.isclass)]

__all__ = [

    # Images
    'show_image',
    'cv2_show_image',
    'cv2_frame_center',
    'cv2_remove_everything_except_square',
    'pillow_resize_image',
    'cv2_resize_image',
    'cv2_sobel_edge_detection',
    'cv2_image_to_pillow',
    'pillow_image_to_cv2',
    'cv2_draw_bounding_boxes',
    'hex_color_to_rgb',
    'get_image_size',
    'get_image_from_url',


    # YOLO
    'normal_bb_coordinates_to_yolo_format',
    'yolo_draw_bbs_path',
    'yolo_draw_single_bb_cv2',


    # Files, folders and I/O
    'load_pickle_file',
    'extract_file_extension',
    'copy_folder',
    'read_txt_file',
    'save_as_pickle_file',
    'write_to_file',
    'read_json',
    'number_of_files_in',
    'get_module_path',
    'get_all_available_import_functions',
    'get_all_available_import_classes',
    'get_current_working_directory',
    'get_changed_file_name',

    # General info
    'get_gpu_memory_info',
    'get_imports',
    'on_windows',
    'get_general_computer_info',
    'get_gpu_info',
    'get_current_screen_dimensions',

    # Plot
    'plot_average_uncertainty',
    'save_plt_plot',
    'plot_lambda',

    # Jupyter notebook
    'expand_jupyter_screen',
    'in_jupyter',
    'play_audio',

    # Formatting
    'scientific_notation',
    'string_to_dict',
    'string_to_list',

    # Classes
    'colors_rgb',
    'colors_hex',
    'Timer',
    'FPS_Timer',

    # Random
    'sturges_rule',
    'to_bins',
    'unfair_coin_flip',
    'init_2d_array',
    'get_grid_coordinates',
    'int_sign',
    'pandas_standardize_df'


    ]