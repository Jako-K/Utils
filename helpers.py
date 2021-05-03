def _imports():
	"""
	Sole purpose is to hide imports
	"""
	import IPython
	import os

	import time
	import datetime

	import torchaudio

	import matplotlib.pyplot as plt
	from PIL import Image
	import pathlib

_imports()


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


def get_gpu_memory_info():
    """ Return the systems total amount of VRAM along with current used/free VRAM"""
    # TODO: check if ´nvidia-smi´ is installed 
    # TODO: Enable multi-gpu setup i.e. cuda:0, cuda:1 ...
    import subprocess as sp
    import os
    
    def get_info(command):
        assert command in ["free", "total"]
        command = f"nvidia-smi --query-gpu=memory.{command} --format=csv"
        info = output_to_list(sp.check_output(command.split()))[1:]
        values = [int(x.split()[0]) for i, x in enumerate(info)]
        return values[0]
        
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    free_vram = get_info("free")
    total_vram = get_info("total")
    return {"free": free_vram, "used": total_vram-free_vram, "total":total_vram}
    

def expand_jupyter_screen(percentage:int = 75):
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
    timer.end()

    print(timer.get_elapsed_time())

    """

    def __init__(self):
        self._start_time = None
        self._elapsed_time = None
        self._unit = "hour/min/sec"

    def start(self):
        self._start_time = time.time()

    def end(self):
        self._elapsed_time = round(time.time() - self._start_time,3)

    def get_elapsed_time(self):

        if self._unit == "hour/min/sec":
            return str(datetime.timedelta(seconds=self._elapsed_time)).split(".")[0] # the ugly bit is just to remove ms
        elif self._unit == "seconds":
            return self._elapsed_time
        elif self._unit == "minutes":
            return self._elapsed_time / 60.0
        elif self._unit == "hours":
            return self._elapsed_time / 3600.0
        elif self._elapsed_time is None:
            return None
        else:
            raise RuntimeError("Should not have gotten this far")

    def set_unit(self, time_unit:str = "hour/min/sec"):
        assert time_unit in ("hour/min/sec", "seconds", "minutes", "hours")
        self._unit = time_unit


def show_image(path:str):
    assert in_jupyter(), "Most be in Jupyer notebook"
    assert os.path.exists(path), "Bad path"
    display(Image.open(path))



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


def save_as_pickle_file(obj, file_name:str, save_path:str=None):
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
    def __init__(self):
        self.blue = (31, 119, 180)
        self.orange = (255, 127, 14)
        self.green = (44, 160, 44)
        self.red = (214, 39, 40)
        self.purple = (148, 103, 189)
        self.brown = (140, 86, 75)
        self.pink = (227, 119, 194)
        self.grey = (127, 127, 127)

    def random_color(self):
        return [random.randint(0, 255) for i in range(3)]

    def is_legal(self, color):
        for color_channel in color:
            if not(0 <= color_channel <= 255):
                return False
        return True
colors_rgb = _ColorRGB()

class _ColorHEX:
    def __init__(self):
        self.blue = "#1f77b4"
        self.orange = "#ff7f0e"
        self.green = "#2ca02c"
        self.red = "#d62728"
        self.purple = "#9467bd" 
        self.brown = "#8c564b"
        self.pink = "#e377c2"
        self.grey =  "#7f7f7f"
colors_hex = _ColorHEX()


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
    import utils.torch_helpers as T
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
    import os\
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