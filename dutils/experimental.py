"""
Description
Stuff that hasn't been tested yet or that I'm on the fence about
"""
import os.path

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

from . import colors as _colors
from . import input_output as _input_output
from . import type_check as _type_check
from . import images as _images
from . import pytorch as _pytorch
from . import system_info as _system_info
from . import videos as _videos

import pkg_resources as _pkg_resources


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
        automatically put Tensors on cuda if it's available.

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


def keep_frames_with_humans(video_path:str, model, batch_size:int=32):

    # Split video into frames (it's probably unwise to keep the frames in RAM, so will put them in temporary folder)
    temp_path = "./temp_image_folder_" + str(_random.getrandbits(128))
    _os.mkdir(temp_path)
    _videos.video_to_images(video_path, temp_path)

    # Create "batches" of paths
    paths = sorted(_glob(_os.path.join(temp_path, "*.png")), key=_os.path.getmtime)
    batch, batches = [], []
    for i, p in enumerate(paths):
        batch.append(p)
        if i and (((i + 1) % batch_size == 0) or ((i + 1) == len(paths))):
            batches.append(batch)
            batch = []

    # inference
    findings = []
    for batch in batches:
        results = model(batch)

        # Check something is detected --> 0, 1 is person and cycle respectively in COCO
        for p in results.pred:
            if p.shape[0] and any(p[:, 5] < 2):
                findings.append(True)
            else:
                findings.append(False)

    # Remove all frames without a human or a cycle
    assert len(findings) == len(paths), "Must be true"
    final_image_paths = [path for i, path in enumerate(paths) if findings[i]]

    # Make the final video"
    save_path = video_path[:-4] + "_reduced.mp4"
    _videos.images_to_video(final_image_paths, save_path)
    _shutil.rmtree(temp_path)


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
    "turn_off_numpy_scientific"
]


