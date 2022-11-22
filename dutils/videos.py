import cv2 as _cv2
import re as _re
import time as _time
import os as _os

from .jupyter_ipython import in_jupyter as _in_jupyter
from . import type_check as _type_check
from . import input_output as _io
from . import images as _images
import subprocess as _subprocess
import platform as _platform
import random as _random

if _in_jupyter():
    from tqdm.notebook import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm


def get_video_info(path:str) -> dict:
    """
    Return general information (FPS, dimensions, etc.) about the video at `path`
    @param path:
    @return: dict with video info
    """
    _type_check.assert_type(path, str)
    _io.assert_path(path)
    if not _io.is_file(path):
        raise ValueError("`path` must point to a file")

    # Info from cv2 # TODO Transfer this to FFMPEG instead of using cv2
    cap = _cv2.VideoCapture(path)
    info = dict(
        video_format = _io.get_file_extension(path).replace(".", ""),
        size = _io.get_file_size(path),
        height = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)),
        width = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)),
        frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)),
        fps = int(cap.get(_cv2.CAP_PROP_FPS)),
    )
    info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
    info["duration_hms"] = _re.sub(r"00[hms] |0(?=[1-9])", "", _time.strftime("%Hh %Mm %Ss", _time.gmtime(info["duration_sec"])))

    # Info from FFMPEG (Could not find a way to get it from cv2)
    if _platform.system() == 'Windows': cmd = ["ffprobe", "-show_streams", path]
    else: cmd = ["ffprobe -show_streams " + path]
    p = _subprocess.Popen(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE, shell=True)
    lines = [str(line.decode('UTF-8', 'ignore')) for line in iter(p.stdout.readline, b'')]

    for line in lines:
        if line.startswith("bit_rate="):
            info["bit_rate"] = line[len("bit_rate="):].strip()
        if line.startswith("codec_long_name="):
            info["codec_long_name"] = line[len("codec_long_name="):].strip()

    return info


def preprocess_video(load_path: str, save_path: str, save_every_nth_frame: int = 3, scale_factor: float = 0.5,
                     rotate_angle: int = 0, fps_out: int = 10, extra_apply: _type_check.FunctionType = None) -> None:
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
    _io.path_exists(load_path)
    legal_formats = [".mp4", ".avi"]
    _type_check.assert_in(_io.get_file_extension(load_path).lower(), legal_formats)
    _type_check.assert_in(_io.get_file_extension(save_path).lower(), legal_formats)

    # Setup
    temp_images = []
    cap = _cv2.VideoCapture(load_path)
    frame_i = -1
    num_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = _tqdm(total=num_frames)

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


def video_to_images(video_path:str, image_folder_path:str) -> None:
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
    _io.path_exists(video_path)
    _io.path_exists(image_folder_path)
    _type_check.assert_in(_io.get_file_extension(video_path).lower(), [".mp4", ".avi"])

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
        save_path = _os.path.join(image_folder_path, str(frame_i)) + ".png"
        successful_save = _cv2.imwrite(save_path, frame)
        if not successful_save:
            raise RuntimeError(f"Failed to save frame {frame_i}, cause unknown")


def images_to_video(image_paths:list, video_save_path:str, fps:int=30, allow_override:bool=False) -> None:
    """
    Make a video in .mp4 format from images at `image_paths` with `fps`.

    @param image_paths: A list with 2 or more images. NOTE: all images must have the shape height, width and colors-channels
    @param video_save_path: The path of the final video. NOTE: this must be suffixed by ".mp4"
    @param fps: Frames per second
    @param allow_override: If true, will override any already existing video located at `video_save_path`
    @return: None
    """

    # Checks
    _type_check.assert_types([image_paths, video_save_path, fps, allow_override], [list, str, int, bool])
    if not allow_override:
        _io.assert_path_dont_exists(video_save_path)
    if not all([_os.path.exists(p) for p in image_paths]):
        raise ValueError("Received one or more bad image paths.")
    if len(image_paths) < 2:
        raise ValueError(f"`image_paths` must contain at least 2 images, received `{len(image_paths)}`")
    if video_save_path[-4:].lower() != ".mp4":
        raise ValueError("The only supported video format is .mp4")
    if fps < 1:
        raise ValueError(f"`fps` cannot be less then 1, received {fps}")

    # Setup video writer and pass it the first frame
    image = _cv2.imread(image_paths[0])
    h, w, c = image.shape
    fourcc = _cv2.VideoWriter_fourcc(*'mp4v')
    video = _cv2.VideoWriter(video_save_path, fourcc, fps, (w, h))
    video.write(image)

    # Pass the rest of the images to `video`
    for path in image_paths[1:]:
        image = _cv2.imread(path)

        # Check image dimensions and close everything if there's an error
        if (h, w, c) != image.shape:
            _cv2.destroyAllWindows()
            video.release()
            raise ValueError(f"Expect all images in `image_paths` to have the same dimensions, but {(h, w, c)} != {image.shape}")

        video.write(image)

    _cv2.destroyAllWindows()
    video.release()


def get_frame_from_video(video_path:str, random_frame:bool=False):
    """
    Select a single frame from `video_path`.

    @param video_path: path pointing to a video file
    @param random_frame: if True, return af random frame, otherwise return the first frame
    @return: image in np.ndarray format
    """
    # Checks
    _type_check.assert_types([video_path, random_frame], [str, bool])
    _io.path_exists(video_path)

    # Setup
    cap = _cv2.VideoCapture(video_path)
    save_index = 0 if not random_frame else _random.randint(0, int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))-1)
    frame_i = -1
    return_frame = None

    # Loop over video to find the frame specified by `save_index`
    while cap.isOpened():
        frame_i += 1
        video_feed_active, frame = cap.read()

        if (save_index == frame_i) or (not video_feed_active):
            return_frame = frame
            cap.release()
            _cv2.destroyAllWindows()
            break

    # Wrap up
    if return_frame is None:
        raise RuntimeError(f"Failed to extract frame, cause unknown")
    return return_frame


__all__ = [
    "get_video_info",
    "preprocess_video",
    "video_to_images",
    "images_to_video",
    "get_frame_from_video"
]
