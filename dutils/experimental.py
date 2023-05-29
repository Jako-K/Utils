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
import PyPDF2 as _PyPDF2
from tkinter import Tcl as _Tcl
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from pytube import YouTube as _Youtube
from pytube.cli import on_progress as _on_progress
import subprocess as _subprocess
from geopy.geocoders import Nominatim as _Nominatim
import pypdf as _pypdf

from . import colors as _colors
from . import input_output as _input_output
from . import type_check as _type_check
from . import images as _images
from . import pytorch as _pytorch
from . import system_info as _system_info
from . import videos as _videos



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
    assert any([numbers ,upper_case ,lower_case ,special_symbol]), "At least one character option need to be active"

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

    file_path = _os.path.dirname(_os.path.abspath(__file__))
    df = _pd.read_csv(file_path + "/_data/wordle.csv")["words"]
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
    """ Primarily used to plot images that has been normalized as tensors through e.g. augmentation"""
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def download_youtube_video(url:str, path_folder:str="./", video_name:str = None, only_video:bool = False, add_progress_bar:bool=True):
    # From the officiel tutorial:
    # https://pytube.io/en/latest/user/quickstart.html
    
    assert _re.search("^https://www\.youtube\.com/watch\?v=", url), "Bad url"
    assert (video_name[-4:].lower() == ".mp4") or (video_name is None), "`video_name` should end on `.mp4` or be `None`"
    if add_progress_bar:
        yt = _YouTube(url, on_progress_callback=_on_progress)
    else:
        yt = _YouTube(url)
    stream = yt.streams.filter(progressive=False, only_video=only_video, file_extension='mp4')
    stream = stream.get_highest_resolution()
    stream.download(path_folder, filename=video_name)


def get_accessible(x):
    """ 
    Return everything accessible from `x`. 
    NOTE: this does not include anything staring with or ending with `_`
    """
    accessible = dir(x)
    accessible = [a for a in accessible if not (a.startswith("_") or a.endswith("_"))]
    return accessible



def read_pdf(pdf_path:str, combine_pages:bool=True) -> _List[str]:
    """ Takes a pdf path and return each a list of pages as strings"""
    file = open(pdf_path, 'rb')
    file_reader = _PyPDF2.PdfFileReader(file)
    pages = [page.extract_text() for page in file_reader.pages]

    if not combine_pages:
        return pages

    pages_combined = ""
    for i, page in enumerate(pages):
        pages_combined += f"\n\n{'#'*75}\n{i}\n{'#'*75}\n\n" 
        pages_combined += page
    return pages_combined


def sorted_lexicographically(l:list):
    return _Tcl().call('lsort', '-dict', l)


def show_frames_as_videos(frames:_List[_np.ndarray], fps:int=10, repeat_n_times:int=1):
    # Every frame in frames should be of shape (H, W, C)
    delay = int(1000 / fps)
    
    if isinstance(frames, _np.ndarray) and (frames.shape == (32, 337, 616, 3)):
        frames = [f for f in frames]
    
    try:
        for i in list(range(len(frames))) * repeat_n_times:
            _cv2.imshow("Video", frames[i][..., ::-1]) # RGB2BGR <-- [..., ::-1]

            if _cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        _cv2.destroyAllWindows()
        
    except Exception:
        _cv2.destroyAllWindows()


def has_windows_line_end(path:str):
    """
    Linux encodes "new line" as `\n` while windows encodes it as `\r\n`.
    This function just make a naive check on the first line contained at `path`
    and check if it ends with `\r\n`. If yes, I'll assume windows-encoding.
    """

    with open(path, 'rb') as f:
        first_line = f.readline()
    return first_line[-2:] == b"\r\n"


def save_numpy_frames_as_video(clip: _np.ndarray, save_path: str, temp_folder: str, fps: int = 12):
    # Check paths
    assert not _os.path.exists(save_path), "Path already exists"
    assert save_path[-4:] == ".mp4", "Only accept mp4 format"
    assert _os.path.exists(temp_folder), "Folder path already exists"
    assert _os.path.isdir(temp_folder), "`temp_folder` must be a folder, but detected something else"
    assert len(glob(temp_folder + "/*")) == 0, "`temp_folder` must be empty"

    # Check `clip`
    assert isinstance(clip, _np.ndarray) and (len(clip.shape) == 4), "Expect a numpy array of shape (frames, height, width, channels)"
    assert all(d >= 1 for d in clip.shape), "All dims must be strictly positive"
    f, h, w, c = clip.shape  # frames x height x width x channels
    assert f > 1, "A video with less than 2 frames?"
    assert c == 3, "Haven't tested anything else then RGB (greyscale should work though)"
    assert (h > 200) and (w > 200), "Assuming a video this small must be a mistake somehow"

    # Save frames temporarily
    # NOTE: The reason for saving to disk first, is because I wanna use ffmpeg for the video creation as opposed to e.g. cv2
    frame_save_paths = []
    for frame_index, frame in enumerate(clip):
        frame_save_path = temp_folder + '/{:06d}'.format(frame_index + 1) + ".png"
        _cv2.imwrite(frame_save_path, frame[:, :, ::-1])  # [:, :, ::-1] --> RGB2BGR
        frame_save_paths.append(frame_save_path)

    # Make video
    command = f'ffmpeg -r {fps} -i "{temp_folder}/%06d.png" -c:v libx264 "{save_path}"' # NOTE has added codex without testing it!!
    ffmpeg_return_str = str(_subprocess.run(command))
    was_successful = "returncode=0" in ffmpeg_return_str

    # Delete all the frames in `temp_folder`
    for frame_path in frame_save_paths:
        _os.remove(frame_path)

    # Assert the video creation was successful
    assert was_successful, f"Something went wrong while using FFMPEG to create the video `{save_path}`.\n" \
                           f"Error message: `{ffmpeg_return_str}`"


def get_file_dateformat():
    return _datetime.today().strftime("%d-%m-%Y_%H-%M-%S")


def thousands_split(number, dk_split=True):
    reformatted = f"{number:,}"
    return reformatted.replace(",", ".") if dk_split else reformatted


# Extract loaction info from lat/long
def lat_long_coord_2_address_info(row):
    # NOTE: This uses an online API under the hood, at this API is rather slow (~2 requests a second)
    geolocator = _Nominatim(user_agent="http")
    row = row[["lat", "lng"]]
    print(row.tolist())
    coord = str(row.tolist())[1:-1]
    location = geolocator.reverse(coord, exactly_one=True)
    return location.raw["address"]


def merge_pdfs(pdf_paths:_List[str], save_path:str="./merged_pdf_result.pdf", can_overwrite:bool=False) -> str:
    """
    >>> pdf_paths = list(glob("./*.pdf"))
    >>> U.experimental.merge_pdfs(pdf_paths)
    'merged_pdf_result.pdf'
    """

    # Checks
    _type_check.assert_types([pdf_paths, save_path, can_overwrite], [list, str, bool])
    [_input_output.assert_path(p) for p in pdf_paths]
    assert save_path.lower().endswith(".pdf"), f"Expected `{save_path}` to end with `.pdf`"
    if not can_overwrite:
        _input_output.assert_path_dont_exists(save_path)

    # Merge pdfs
    merger = _pypdf.PdfMerger()
    for pdf in pdf_paths:
        merger.append(pdf)
    
    # Wrap up
    merger.write(save_path)
    merger.close()
    return save_path


class TextSearcher:
    """
    # EXAMPLE
    >> searcher = TextSearcher()
    >> to_search_through = ['Defines expansion with a kernel', 'Random forest improves performance']
    
    >> indexes, scores, texts = searcher.sematic_search(to_search_through, "Random forest score", n=-1)
    >> indexes, scores, texts
    ([0, 1], [-0.08474002, 0.7031414], ['Defines expansion with a kernel', 'Random forest improves performance'])
    
    >> indexes, scores, texts = searcher.regex_search(to_search_through, "Random forest.*performance")
    >> indexes, texts
    ([1], ['Random forest improves performance'])
    
    """
    
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1"):
        # Checks
        valid_model_names = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1", "all-MiniLM-L12-v2", "multi-qa-distilbert-cos-v1", "all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-mpnet-base-v2", "paraphrase-albert-small-v2", "paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-MiniLM-L3-v2", "distiluse-base-multilingual-cased-v1", "distiluse-base-multilingual-cased-v2"]
        _type_check.assert_type(model_name, str)
        if model_name not in valid_model_names:
            raise ValueError(f"`{model_name}` is not valid. Choose one of these: `{valid_model_names}`")
        
        # Define model and scoring function
        try:
            import sentence_transformers
        except ModuleNotFoundError:
            raise RuntimeError("Could not detect `sentence_transformers`. use `pip3 install sentence-transformers` to continue")
        self.model = sentence_transformers.SentenceTransformer(model_name).cuda()
        self.score_function = sentence_transformers.util.cos_sim #util.dot_score
    
    def regex_search(self, text_strings:list, search_query:str, case_sensitive:bool=False, dont_return_text:bool=False):
        # Checks
        _type_check.assert_types([text_strings, search_query, case_sensitive, dont_return_text], [list, str, bool, bool])
        _type_check.assert_list_slow(text_strings, str)
        
        # Do matches
        matches_indexes_booleans = grep(text_strings, search_query, case_sensitive=case_sensitive, return_booleans=True)
        matches_indexes = [i for i, is_match in enumerate(matches_indexes_booleans) if is_match]
        matches_text = [text_strings[i] for i in matches_indexes]
        if dont_return_text:
            return matches_indexes
        return matches_indexes, matches_text
    
    
    def sematic_search(self, text_strings:list, search_query:str, n:int=None, min_score:float=None, dont_return_text:bool=False):
        # Checks
        _type_check.assert_types([text_strings, search_query, n, min_score], [list, str, int, float], [0, 0, 1, 1])
        _type_check.assert_list_slow(text_strings, str)
        if min_score is not None:
            assert -1.0 <= min_score <= 1.0
        if n is None:
            n = 1 if (5 > len(text_strings)) else 5
        elif n == -1:
            n = len(text_strings)
        else:
            assert n < len(text_strings)
        
        # Get embeddings for text_string and search_query
        query_embedding = self.model.encode(search_query)
        passage_embedding = self.model.encode(text_strings)
        
        # Similarity scoring
        similarities = self.score_function(query_embedding, passage_embedding)
        similarities = similarities.squeeze(0).numpy()
        if min_score:
            similarities[similarities < min_score] = -_np.inf # Set all scores below the threshold to 0, this will mean they are included from matching
        
        # Extract the top `n` findings
        top_n_indexes = [int(i) for i in list(_np.argsort(similarities)[-n:]) if similarities[i] != -_np.inf]
        top_n_scores  = [similarities[i] for i in top_n_indexes]
        top_n_texts   = [text_strings[i] for i in top_n_indexes]
        
        if dont_return_text:
            return top_n_indexes, top_n_scores
        return top_n_indexes, top_n_scores, top_n_texts


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
    "download_youtube_video",
    "get_accessible",
    "read_pdf",
    "sorted_lexicographically",
    "show_frames_as_videos",
    "has_windows_line_end",
    "save_numpy_frames_as_video",
    "get_file_dateformat",
    "thousands_split",
    "lat_long_coord_2_address_info",
    "merge_pdfs",
    "TextSearcher",
]


