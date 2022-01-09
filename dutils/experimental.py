"""
Description
Stuff that hasn't been tested yet or that I'm on the fence about
"""

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

from . import colors as _colors
from . import input_output as _input_output
from . import type_check as _type_check
from . import images as _images
from . import pytorch as _pytorch

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


def preprocess_video(load_path:str, save_path:str, save_every_nth_frame:int=3, scale_factor:float = 0.5,
                     rotate_angle:int=0, fps_out:int=10, extra_apply:_type_check.FunctionType=None):
    """
    Load video located at `load_path` for processing: Reduce FPS by `save_every_nth_frame`,
    resize resolution by `scale_factor` and clockwise rotation by `rotate_angle` .
    The modified video is saved at `save_path` with a FPS of `fps_out`
    NOTE: The frames used to reconstruct the video is kept in memory during processing, which may be problematic for larger videos

    EXAMPLE:
    >> preprocess_video("video_in.avi", "video_out.mp4", 10, 0.35, -90, 10)

    @param load_path: Load path to video for processing. Must be ".mp4" or ".avi"
    @param save_path: Save path to processed video. Must be ".mp4" or ".avi"
    @param save_every_nth_frame: Essentially a downscaling factor e.g. 3 would result in a video containing 1/3 of the frames
    @param scale_factor: Determine the image size e.g. 0.25 would decrease the resolution by 75%
    @param rotate_angle: Clockwise rotation of the image. must be within [0, 90, -90, 180, -180, -270, 270]
    @param fps_out: The frame rate of the processed video which is saved at `save_path`
    @param extra_apply: and extra function which can be applied to the image at the very end e.g. for cropping, noise etc.
    @return: None
    """

    # Checks
    _type_check.assert_types(
        to_check=[load_path, save_path, save_every_nth_frame, scale_factor, rotate_angle, fps_out],
        expected_types=[str, str, int, float, int, int]
    )
    _input_output.path_exists(load_path)
    legal_formats = [".mp4", ".avi"]
    _type_check.assert_in(_input_output.get_file_extension(load_path).lower(), legal_formats)
    _type_check.assert_in(_input_output.get_file_extension(save_path).lower(), legal_formats)
    
    # Setup
    temp_images = []
    cap = _cv2.VideoCapture(load_path)
    frame_i = -1

    # Prepare all frames for modified video
    while cap.isOpened():
        frame_i += 1
        video_feed_active, frame = cap.read()

        if not video_feed_active:
            cap.release()
            _cv2.destroyAllWindows()
            break

        if frame_i%save_every_nth_frame != 0: # Only process every n'th frame.
            continue
            
        resized = _images.ndarray_resize_image(frame, scale_factor)
        rotated = _images.rotate_image(resized, rotate_angle)
        final = rotated if extra_apply is None else extra_apply(rotated)
        temp_images.append(final)


    # Combine processed frames to video
    height, width, _ = temp_images[0].shape
    video = _cv2.VideoWriter(save_path, 0, fps_out, (width,height))

    for image in temp_images:
        video.write(image)

    _cv2.destroyAllWindows()
    video.release()


def normal_dist(x , mu , std):
    return  1/(std*_np.sqrt(2*_np.pi)) * _np.exp(-0.5 * ((x - mu)/std)**2)


def get_module_version(module_name:str):
    _type_check.assert_type(module_name, str)
    return _pkg_resources.get_distribution(module_name).version


def video_to_images(video_path: str, image_folder_path: str) -> None:
    """
    Break a video down into individual frames and save them to disk.

    EXAMPLE:
    >> video_to_images("./video_in.MP4", "./frames_folder")

    @param video_path: Load path to video for processing. Must be ".mp4" or ".avi"
    @param image_folder_path: Path to folder where the frames are to be saved
    @return: None
    """

    # Checks
    _type_check.assert_types(to_check=[video_path, image_folder_path], expected_types=[str, str])
    _input_output.path_exists(video_path)
    _input_output.path_exists(image_folder_path)
    _type_check.assert_in(_input_output.get_file_extension(video_path).lower(), [".mp4", ".avi"])

    # Extract and save individual frames
    cap = _cv2.VideoCapture(video_path)
    frame_i = -1
    while cap.isOpened():
        frame_i += 1
        video_feed_active, frame = cap.read()

        if not video_feed_active:
            cap.release()
            _cv2.destroyAllWindows()
            break

        # Save frame
        save_path = _os.path.join(image_folder_path, str(frame_i)) + ".jpg"
        successful_save = _cv2.imwrite(save_path, frame)
        if not successful_save:
            raise RuntimeError(f"Failed to save frame {frame_i}, cause unknown")


def IoU_score(p1:list, p2:list) -> float:
    """
    Calculates the intersection over union (IoU) between the two bounding boxes `p1` and `p2`

    @param p1: Bounding box list: [up_left_x, up_left_y, low_right_x, low_right_y]
    @param p2: Bounding box list: [up_left_x, up_left_y, low_right_x, low_right_y]
    @return: IoU score
    """

    x1_hat, y1_hat, x2_hat, y2_hat = p1
    x1, y1, x2, y2 = p2

    dx1 = x1_hat - x1
    dy1 = y1_hat - y1
    dx2 = x2_hat - x2
    dy2 = y2_hat - y2
    w = abs(x1_hat-x2_hat)
    h = abs(y1_hat-y2_hat)

    ix1 = max(x1, x1_hat) #if x1_hat <= x2 else None
    iy1 = max(y1, y1_hat) #if y1_hat <= y2 else None

    dx2 = x2_hat - x2
    dy2 = y2_hat - y2
    ix2 = min(x2, x2_hat) #if x2_hat >= x1 else None
    iy2 = min(y2, y2_hat) #if y2_hat >= y1 else None

    print(ix1, iy1, ix2, iy2)
    if None in [ix1, iy1, ix2, iy2]:
        return 0,0,0,0

    return ix1, iy1, ix2, iy2


def get_test_image(load_type:str="unchanged"):
    dutils_path = _os.path.dirname(__file__)
    image_path = _os.path.join(dutils_path, "_unit_tests/dragon.jpg")
    return _images.load_image(image_path, load_type)




# import scipy.io
# --> mat = scipy.io.loadmat('file.mat')


# CV_RGB2HLS
"""
import cv2
import numpy as np

img = cv2.imread('messi5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(img,100,200)

#sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)


cv2.imshow("Original Image", img)
cv2.imshow("Canny", img_canny)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""


__all__ = [
    "show_dicom",
    "load_dicom",
    "show_dicom_ndarray",
    "load_unspecified",
    "bucket_continuous_feature",
    "stratified_folds",
    "DataLoaderDevice",
    "preprocess_video",
    "get_module_version",
    "normal_dist",
    "get_test_image"
]


