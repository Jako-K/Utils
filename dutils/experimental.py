"""
Description
Stuff that hasn't been tested yet or I'm on the fence about
"""
import gc as _gc
from typing import List as _List

import torch as _torch
import numpy as _np
import warnings as _warnings
import pandas as _pd
import seaborn as _sns
import matplotlib.pylab as _plt
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from torch.utils.data import DataLoader as _DataLoader
import pydicom as _dicom
import cv2 as _cv2
import os as _os
from glob import glob as _glob
import re as _re
import random as _random
import shutil as _shutil
import pathlib as _pathlib
import pkg_resources as _pkg_resources
import json as _json

from . import colors as _colors
from . import input_output as _input_output
from . import type_check as _type_check
from . import images as _images
from . import pytorch as _pytorch
from . import system_info as _system_info
from . import videos as _videos
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from pytube import YouTube as _Youtube

def load_dicom(path:str):
    # Checks
    _input_output.assert_path(path)
    if _input_output.get_file_extension(path)[-4:] != ".dcm":
        raise ValueError("Expected `.dcom` extension, but received something else")

    return _dicom.dcmread(path).pixel_array


def show_dicom(path:str):
    dicom_image = load_dicom(path)
    _plt.imshow(dicom_image, cmap="bone")
    _plt.axis("off")


def show_dicom_ndarray(dicom_image):
    _plt.imshow(dicom_image, cmap="bone")
    _plt.axis("off")


def load_unspecified(path:str):
    """
    Try and load whatever is being passed: image.jpg, sound.wav, text.txt etc.
    and return it in an appropriate format.
    The reason why this could be nice, is that if you don't know where some specific loader is e.g. image_load
    This would just automatically find it for you, or say that it just don't exists

    # TODO is this a good idea?

    # image [.jpg, .png] as nd.array
    # dicom [.dcm] as nd.array
    # text [.txt, .json] as string
    # sound [wav] as nd.array
    # video [mp4, avi] as list[nd.array images]

    """
    raise NotImplementedError("")


def bucket_continuous_feature(data:_np.ndarray, bins:int, plot:bool=True) -> _pd.DataFrame:
    """
    Bucket `data` into `bins` number of bins. Use `plot=True` for visual inspection

    @param data: 1D numpy array with numbers
    @param bins: Number of buckets/bins
    @param plot: Visualise buckets
    @return: bucket_labels:np.ndarray
    """

    # Checks
    _type_check.assert_types([data, bins, plot], [_np.ndarray, int, bool])
    _type_check.assert_comparison_number(bins, 1, ">=", "bins")
    if len(data.shape) != 1:
        raise ValueError(f"Expected `data` to be of shape (n,) but received `{data.shape}`")
    # TODO: check if data.dtype is numeric

    df = _pd.DataFrame( _np.zeros(len(data)), columns=["bucket_label"], dtype=int )
    df["value"] = data

    # Bucket the data
    df["bucket_label"] = _pd.cut(data, bins=bins, labels=False)

    if plot:
        buckets_density = df.groupby("bucket_label").count()["value"] / len(df)
        _plt.figure(figsize=(15, 7))
        _plt.bar(range(len(buckets_density)), buckets_density)
        _plt.title("Bucket distribution")
        _plt.xlabel("Bucket label")
        _plt.ylabel("Density")

    return df["bucket_label"].to_frame()


def stratified_folds(labels:_np.ndarray, folds:int, seed:int=12, as_pandas:bool=False):
    # TODO Add some stats such as label distribution and train/valid ratio
    """
    Stratified k-fold split.

    @param labels: 1D numpy array used for stratification, must be discrete.
    @param folds: Number of folds
    @param seed: seed used in sklearn's StratifiedKFold
    @param as_pandas: return a pandas DataFrame
    @return: pandas DataFrame with fold information if `as_pandas`
             else a dict with {folds:int = (train_idx:np.ndarray, valid_idx:np.ndarray) ...}
    """

    # Assign stratified folds
    kf = _StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    iterator = kf.split(X=_np.arange(len(labels)), y=labels) # X=labels was just the easiest, it gets discarded anyway.

    if not as_pandas:
        return {fold:(train_idx, valid_idx) for fold, (train_idx, valid_idx) in enumerate(iterator)}

    # If as pandas
    df = _pd.DataFrame(_np.zeros(len(labels)), columns=["fold"], dtype=int)
    for i, (_, fold_idx) in iterator:
        df.loc[fold_idx, 'fold'] = i

    return df


class DataLoaderDevice:
    def __init__(self, dl:_DataLoader, device:str=None):
        """
        A simple wrapper class for pytorch's DataLoader class which
        automatically puts Tensors on cuda if it's available.

        @param dl: Pytorch DataLoader
        @param device: Device to put tensors on, if None `get_device()` is used for detection
        """
        _type_check.assert_types([dl, device], [_DataLoader, str], [0, 1])

        self.dl = dl
        self.device = device if device is not None else _pytorch.get_device()

    def _to_device_preprocess(self, *args):
        """ Put everything of type `Tensor` on cuda if it's available and on cpu if not """
        return [a.to(self.device) if isinstance(a, _torch.Tensor) else a for a in args]

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self._to_device_preprocess(b)


def normal_dist(x , mu , std):
    return  1/(std*_np.sqrt(2*_np.pi)) * _np.exp(-0.5 * ((x - mu)/std)**2)


def get_module_version(module_name:str):
    _type_check.assert_type(module_name, str)
    return _pkg_resources.get_distribution(module_name).version


def get_test_image(load_type:str="unchanged", as_tensor:bool=False):
    # checks
    _type_check.assert_types([load_type, as_tensor], [str, bool])

    dutils_path = _os.path.dirname(__file__)
    image_path = _os.path.join(dutils_path, "_unit_tests/dragon.jpg")
    image = _images.load_image(image_path, load_type="RGB" if as_tensor else "unchanged")
    return _torch.tensor(image).permute(2,0,1) if as_tensor else image


def turn_off_numpy_scientific():
    _np.set_printoptions(suppress=True)


def read_yolo_annotation_file(path: str, auto_clip: bool = False) -> list:
    """
    Read a yolo annotation file. Expects a .txt file build from lines like:
    class_labal_int x_mid y_mid box_width box_height

    Return a tuple with a list of BBS and a list of labels
    ([2, 0], [[0.6445, 0.4383, 0.4743, 0.8676], [0.6013, 0.6334, 0.362, 0.4665]])
    """
    as_single_str = _input_output.read_file(path).strip()
    if not as_single_str:
        return None

    anno_as_lines = as_single_str.split("\n")
    anno_as_lists = [[round(float(n), 4) for n in l.split(" ")] for l in anno_as_lines]
    return _torch.tensor(anno_as_lists)


def ndarray_grey_to_rgb(image:_np.ndarray):
    """ Copy the greyscale 3 times to form R, G and B. Shape change: (H, W) --> (H, W, 3)"""
    _images.assert_ndarray_image(image)
    return _cv2.merge([image] * 3)


def ndarray_save_image(image:_np.ndarray, save_path:str, BGR2RGB:bool=True) -> None:
    _type_check.assert_types([image, save_path, BGR2RGB], [_np.ndarray, str, bool])
    if BGR2RGB:
        image = _images.ndarray_bgr2rgb(image)
    _cv2.imwrite(save_path, image)


def xywhn2xyxy(label, x, y, w, h, image_height, image_width):
    # Convert xywhn to cartesian coordinates
    x_left = int((x - w / 2) * image_width)
    x_right = int((x + w / 2) * image_width)
    y_top = int((y - h / 2) * image_height)
    y_bottom = int((y + h / 2) * image_height)

    # Clip if close to the edges
    if x_left < 0: x_left = 0
    if x_right > (image_width - 1): x_right = image_width - 1
    if y_top < 0: y_top = 0
    if y_bottom > (image_height - 1): y_bottom = image_height - 1
    
    return int(label), x_left, y_top, x_right, y_bottom



def grep(strings:list, pattern, case_sensitive:bool=True, return_booleans:bool=False):
    # Checks
    _type_check.assert_types([strings, case_sensitive, return_booleans], [list, bool, bool])
    try:
        _re.compile(pattern)
    except _re.error as e:
        raise ValueError(f"Received a bad regex pattern. Failed with: `{pattern}`")
    
    return_values = []
    for string in strings:
        _type_check.assert_type(string, str)
        found_pattern = _re.search(pattern, string, flags=0 if case_sensitive else _re.IGNORECASE)
        if return_booleans:
            return_values.append(found_pattern is not None)
        elif found_pattern:
            return_values.append(string)
            
    return return_values            


def get_folder_memory_use(path, unit:str="MB", only_basename:bool=True):
    # Checks
    _input_output.assert_path(path)
    _type_check.assert_types([unit, only_basename], [str, bool])
    
    # Units 
    #   As i understand it, the whole 1024 vs 1000 is just a legacy from harddrive manufacturers 
    #   wanting to artfically inflate memory capacity by refering to kilo in metric terms i.e. kilo=1000
    #   Whereas, others thought it would be better use something more suitable for computers i.e. kilo=2^10=1024
    unit_mapping = {"Byte":1, "KB":1024, "KiB":1000, "MB":1024**2, "MiB":1000**2, "GB":1024**3, "GiB":1000**3}
    _type_check.assert_in(unit, list(unit_mapping.keys()))
    conversion_unit = unit_mapping[unit]
    
    # Loop over folder and calculate memory footprint
    memory_use = {}
    for path in _pathlib.Path(path).rglob('*'):
        if only_basename:
            memory_use[path.name] = round(os.path.getsize(path) / conversion_unit, 3)
        else:
            memory_use[os.path.abspath(path)] = round(os.path.getsize(path) / conversion_unit, 3)
    
    memory_use["folder_total"] = round(sum(memory_use.values()), 3)
    memory_use_sorted = dict(sorted(memory_use.items(), key= lambda x:x[1], reverse = True))
    
    return memory_use_sorted


def clear_cuda():
    _gc.collect()
    _torch.cuda.empty_cache()


def get_password(length:int=12, numbers:bool=True, upper_case:bool=True, lower_case:bool=True, special_symbol:bool=True):
    drawing_pool = []
    if numbers: 
        drawing_pool += [str(n) for n in range(10)]
    if upper_case:
        drawing_pool += [l for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    if lower_case:
        drawing_pool += [l for l in "abcdefghijklmnopqrstuvwxyz"]
    if special_symbol:
        drawing_pool += "!@#$&=?"
    password = [_random.choice(drawing_pool) for _ in range(length)]
    return "".join(password)


def date_range(start_date:_datetime, end_date:_datetime, date_format:str="%Y-%m-%d"):
    day_diff = range((end_date-start_date).days + 1)
    return [(start_date + _timedelta(days=x)).strftime(date_format) for x in day_diff]


def get_wordle_options(greens:str=".....", yellow_rows:list=["....."], greys:str=""):
    """
    >> get_wordle_options("...e.", [".e...", "fra.."], "hlonkpils")
    ['abbey', 'abyed', 'acred', 'acted', 'added', ...]
    """
    # Checks
    _type_check.assert_types([greens, yellow_rows, greys], [str, list, str])
    assert (len(greens) == 5) and all(len(r) == 5 for r in yellow_rows), "the length of `grens` and each row in `yellow_rows` by be exatcly 5"


    letters_not_on_place = ["" for i in range(5)]
    for yellow_row in yellow_rows:
        for i, letter in enumerate(yellow_row):
            if letter != ".":
                letters_not_on_place[i] += letter
    
    search_query = "^"
    for i, green_letter in enumerate(greens):
        if green_letter != ".":
            search_query += green_letter
        else:
            search_query += f"[^{letters_not_on_place[i] + greys}]"
    search_query += "$"

    df = _pd.read_csv("./_data/wordle.csv")["words"]
    results = df[df.str.contains(search_query)]
    return results.tolist()
    

def ipynb_2_py(path:str) -> dict:
    """ Converts `path` from jupyter notebook format (.ipynb) to regular python (.py) """

    _input_output.assert_path(path)
    if path[-6:] != ".ipynb":
        raise ValueError("Expected .ipynb file extension, but received something else")
    
    with open(path) as f:
        data = _json.load(f)["cells"]
    
    code_base = ""
    hashtag_line = "#"*100 + "\n"

    for i, cell in enumerate(data):
        if cell["cell_type"] == "markdown":
            code_base += f"\n\n{hashtag_line}{''.join(cell['source'])}\n{hashtag_line}\n"

        elif cell["cell_type"] == "code":
            code_base += ''.join(cell['source'])

    return code_base
    

def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def download_youtube_video(url:str, save_path_folder:str="./", video_name:bool = None):
    # From the officiel tutorial:
    # https://pytube.io/en/latest/user/quickstart.html
    
    assert _re.search("^https://www\.youtube\.com/watch\?v=.{11}$", url), "Bad url"
    assert (video_name[-4:].lower() == ".mp4") or (video_name is None), "`video_name` should end on `.mp4` or be `None`"
    yt = _Youtube(url)
    stream = yt.streams.filter(progressive=False, only_video=True, file_extension='mp4')[0]
    stream.download(save_path_folder, filename=video_name)


__all__ = [
    "show_dicom",
    "load_dicom",
    "show_dicom_ndarray",
    "load_unspecified",
    "bucket_continuous_feature",
    "stratified_folds",
    "DataLoaderDevice",
    "get_module_version",
    "normal_dist",
    "get_test_image",
    "turn_off_numpy_scientific",
    "read_yolo_annotation_file",
    "xywhn2xyxy",
    "grep",
    "get_folder_memory_use",
    "clear_cuda",
    "get_password",
    "date_range",
    "get_wordle_options",
    "ipynb_2_py",
    "inverse_normalize",
    "download_youtube_video"
]


