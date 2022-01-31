import cv2 as _cv2
import re as _re
import time as _time
from tqdm.notebook import tqdm as _tqdm_notebook
from tqdm import tqdm as _tqdm
import os as _os

from .jupyter_ipython import in_jupyter as _in_jupyter
from . import type_check as _type_check
from . import input_output as _input_output
from . import images as _images


def get_video_info(path:str) -> dict:
    """  """
    _type_check.assert_type(path, str)
    _input_output.assert_path(path)
    if not _input_output.is_file(path): raise ValueError("`path` most point to a file")

    cap = _cv2.VideoCapture(path)

    info = dict(
        video_format = _input_output.get_file_extension(path).replace(".", ""),
        size = _input_output.get_file_size(path),
        height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)),
        width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)),
        frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)),
        frames_per_sec = int(cap.get(_cv2.CAP_PROP_FPS)),
    )
    info["duration_sec"] = round(info["frame_count"] / info["frames_per_sec"], 2)
    info["duration_hms"] = _re.sub(r"00[hms] |0(?=[1-9])", "", _time.strftime("%Hh %Mm %Ss", _time.gmtime(info["duration_sec"])))
    return info


def preprocess_video(load_path: str, save_path: str, save_every_nth_frame: int = 3, scale_factor: float = 0.5,
                     rotate_angle: int = 0, fps_out: int = 10, extra_apply: _type_check.FunctionType = None):
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
    num_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = _tqdm_notebook(total=num_frames) if _in_jupyter() else _tqdm(total=num_frames)

    # Prepare all frames for modified video
    while cap.isOpened():
        progress_bar.update(1)
        frame_i += 1
        video_feed_active, frame = cap.read()

        if not video_feed_active:
            cap.release()
            _cv2.destroyAllWindows()
            break

        if frame_i % save_every_nth_frame != 0:  # Only process every n'th frame.
            continue

        resized = _images.ndarray_resize_image(frame, scale_factor)
        rotated = _images.rotate_image(resized, rotate_angle)
        final = rotated if extra_apply is None else extra_apply(rotated)
        temp_images.append(final)
    progress_bar.close()

    # Combine processed frames to video
    height, width, _ = temp_images[0].shape
    video = _cv2.VideoWriter(save_path, 0, fps_out, (width, height))

    for image in temp_images:
        video.write(image)

    _cv2.destroyAllWindows()
    video.release()


def video_to_images(video_path: str, image_folder_path: str) -> None:
    """
    Break a video down into individual frames and save them to disk.

    EXAMPLE:
    >> video_to_images("./video_in.MP4", "./frames_folder")

    @param video_path: video load path for processing. Must be ".mp4" or ".avi"
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


def cv2_video_frame_count(cap:_cv2.VideoCapture):
    _type_check.assert_type(cap, _cv2.VideoCapture)
    return int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))


__all__ = [
    "get_video_info",
    "preprocess_video",
    "video_to_images",
    "cv2_video_frame_count"
]
