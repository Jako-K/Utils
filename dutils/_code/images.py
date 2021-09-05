# TODO: Add unit tests

import numpy as _np
import type_check as _type_check
from PIL import Image as _Image
import requests as _requests
import cv2 as _cv2
import warnings as _warnings

import colors as _colors
import input_output as _input_output
import jupyter_ipython as _jupyter


def is_ndarray_greyscale(image: _np.ndarray):
    """ Try and determine if `image` is greyscale/has only 1 channel """
    _type_check.assert_type(image, _np.ndarray)
    if len(image.shape) == 2:
        return True
    elif (len(image.shape) == 3) and (len(image.shape[2]) == 1):
        return True
    else:
        return False


def assert_ndarray_image(image: _np.ndarray):
    """ Function description: Try and ensure that `image` is legitimate """
    _type_check.assert_type(image, _np.ndarray)
    is_greyscale = True if len(image.shape) == 2 else False

    if len(image.shape) != (2 if is_greyscale else 3):
        ValueError(f"Expected 3 dimensional shape for non-greyscale image (dim1 x dim2 x channels). "
                   f"Received shape `{image.shape}`")
    if image.min() < 0:
        ValueError("Detected a negative value in np.ndarray image, this is not allowed.")

    if image.dtype != 'uint8':
        raise TypeError(f"Expected `image` to be of dtype uint8, but received `{image.dtype}`. "
                        f"If `image` is in 0.0-1.0 float format, "
                        "try casting it with: `image = (image * 255).astype(np.uint8)`.")

    if not is_greyscale:
        num_channels = image.shape[2]
        if num_channels not in [3, 4]:
            ValueError(f"Received an unknown number of color channels: `{num_channels}`. "
                       f"Accepted number of channels are 1, 3 and 4 (Greysacle, RGB and RGBA respectively) ")


def pillow_resize_image(image: _Image.Image, resize_factor: float = 1.0):
    """
    Function description:
    Resize `image` according to `resize_factor`

    @param image: Image in PIL.Image format
    @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
    @return: Resized image in PIL.Image format
    """
    _type_check.assert_types([image, resize_factor], [_Image.Image, float])
    width = int(image.size[0] * resize_factor)
    height = int(image.size[1] * resize_factor)
    return image.resize((width, height), resample=0, box=None)


def pillow_image_to_ndarray(image: _Image.Image, RGB2BGR: bool = True):
    """ Convert PIL image to ndarray. `RGB2BGR` is there for cv2 compatibility """
    # Tests
    _type_check.assert_types([image, RGB2BGR], [_Image.Image, bool])
    as_ndarray = _np.asarray(image)

    return as_ndarray if not RGB2BGR else _cv2.cvtColor(as_ndarray, _cv2.COLOR_RGB2BGR)


def image_size_from_path(path:str, WxH=True):
    """ Return image size in width x height or vice versa """
    _input_output.assert_path(path)
    _type_check.assert_type(WxH, bool)

    height, width = _cv2.imread(path).shape[:2]
    return (width, height) if WxH else (height, width)


def get_image_from_url(url: str, return_type: str = "cv2"):
    """
    Fetch and return image in the format specified by `return_type` from URL.
    Note that no checks are performed on the `url`
    """
    # Checks
    _type_check.assert_types([url, return_type], ["str", "str"])
    if return_type not in ["pillow", "cv2"]:
        raise ValueError(f"Expected `return_type` to be in ['pillow', 'cv2'], but received `{return_type}`")

    # Download, open and return image
    image = _Image.open(_requests.get(url, stream=True).raw)
    return image if (return_type == "pillow") else _colors.convert_color(image, "rgb")


def ndarray_image_to_pillow(image: _np.ndarray, BGR2RGB: bool = True):
    """ Convert ndarray image to PIL. `BGR2RGB` is there for cv2 compatibility """
    # Tests
    _type_check.assert_types([image, BGR2RGB], [_np.ndarray, bool])
    assert_ndarray_image(image)

    if BGR2RGB:
        image = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
    return _Image.fromarray(image)


def ndarray_resize_image(image: _np.ndarray, resize_factor: float):
    """
    Function description:
    Resize `image` according to `resize_factor`

    @param image: Image in np.ndarray format
    @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
    @return: Resized image in ndarray format
    """
    # Checks
    _type_check.assert_types([image, resize_factor], [_np.ndarray, float])
    assert_ndarray_image(image)
    if resize_factor < 0:
        ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    new_width = int(image.shape[1] * resize_factor)
    new_height = int(image.shape[0] * resize_factor)
    return _cv2.resize(image, (new_width, new_height), interpolation=_cv2.INTER_AREA)


def show_image( path: str, resize_factor: float = 1.0):
    """
    Function description:
    Display a single image from `path` (works in standard python and jupyter environment)

    @param path: Path to an image
    @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
    """
    # TODO Include support for URL image path

    # Checks
    _type_check.assert_types([path, resize_factor], [str, float])
    _input_output.assert_path(path)
    if resize_factor < 0:
        ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    # If inside a jupyter environment Pillow is ued to show the image, otherwise cv2.
    if _jupyter.in_jupyter():
        image = _Image.open(path)
        image = pillow_resize_image(image, resize_factor)
        display(image)
    else:
        image = _cv2.imread(path)
        image = ndarray_resize_image(image, resize_factor)
        show_ndarray_image(image)


def show_ndarray_image(image: _np.ndarray, resize_factor: float = 1.0, name: str = "", BGR2RGB:bool = False):
    """
    Function description:
    Display a single image from a `np.ndarray` source (works in standard python and jupyter environment)

    @param image: Image in np.ndarray format
    @param resize_factor: Rescale factor in percentage, `scale_factor` < 0
    @param name: window name used in `cv2.imshow()` (Has no effect when inside jupyter notebook environment)
    @param BGR2RGB: Swap the blue and red color channel i.e. BGR --> RGB
    """

    # Checks
    _type_check.assert_types([image, resize_factor, name, BGR2RGB], [_np.ndarray, float, str, bool])
    assert_ndarray_image(image)
    if resize_factor < 0:
        ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    if BGR2RGB:
        image = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)

    # If inside a jupyter environment Pillow is ued to show the image, otherwise cv2.
    if _jupyter.in_jupyter():
        image = ndarray_image_to_pillow(image, BGR2RGB=False)
        image = pillow_resize_image(image, resize_factor)
        display(image)
    else:
        image = ndarray_resize_image(image, resize_factor)
        _cv2.imshow(name, image)
        _cv2.waitKey(0)
        _cv2.destroyAllWindows()


def ndarray_image_center(image: _np.ndarray, WxH: bool = True):
    """
    Function description:
    Calculate the center of an image

    @param image: Image in np.ndarray format
    @param WxH: If true width x height is returned, if false height x width
    @return: center of frame (either in WxH or HxW)
    """
    # Tests
    _type_check.assert_types([image, WxH], [_np.ndarray, bool])
    assert_ndarray_image(image)

    h, w = image.shape[:2]
    return (w // 2, h // 2) if WxH else (h // 2, w // 2)


def cv2_cutout_square(image: _np.ndarray, p1:tuple, p2:tuple, inverse:bool = False):
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
    _type_check.assert_types([image, p1, p2, inverse], [_np.ndarray, tuple, tuple, bool])
    _type_check.assert_list_slow(p1, int, 2)
    _type_check.assert_list_slow(p2, int, 2)
    assert_ndarray_image(image)

    ## Check if bb coordinates a reasonable
    h, w = image.shape[:2]
    if not ((0 < p1[0] <= w) and (0 < p2[0] <= w) and (0 < p2[1] <= h) and (0 < p2[1] <= h)):
        raise ValueError("BB coordinates `p1` and `p2` violate image width/height and/or are <= 0")

    # Make the mask (the whole `if inverse else` is just to define what is black and white)
    mask = _np.ones(image.shape[:2], dtype="uint8") * (0 if inverse else 255)
    _cv2.rectangle(mask, p1, p2, (255 if inverse else 0), -1)

    masked_image = _cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def cv2_sobel_edge_detection(image: _np.ndarray, blur_kernel_size: int = 5,
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
    _type_check.assert_types([image, blur_kernel_size, horizontal, vertical], [_np.ndarray, int, bool, bool])
    assert_ndarray_image(image)
    if not (horizontal or vertical):
        raise ValueError("`horizontal` and `vertical` cannot both be False")

    # Preprocess
    gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
    blurred = _cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Sobel
    horizontal_sobel, vertical_sobel = None, None
    if horizontal:
        horizontal_sobel = _cv2.Sobel(blurred, 0, 1, 0, _cv2.CV_64F)
    if vertical:
        vertical_sobel = _cv2.Sobel(blurred, 0, 0, 1, _cv2.CV_64F)

    # Looks more mysterious then it is. Its simple away of dealing with all possible values of `horizontal` and `vertical`
    edges = _cv2.bitwise_or(horizontal_sobel if horizontal else vertical,
                            vertical_sobel if vertical else horizontal_sobel)
    return edges


def cv2_draw_bounding_boxes(image: _np.ndarray, p1: tuple, p2: tuple, label: str = None,
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
    _type_check.assert_types(
        to_check=[image, p1, p2, label, conf, color, line_thickness],
        expected_types=[_np.ndarray, tuple, tuple, str, float, object, int],
        allow_nones=[0, 0, 0, 1, 1, 1, 0, 0]
    )
    _type_check.assert_list_slow(p1, int, 2)
    _type_check.assert_list_slow(p2, int, 2)

    ## Check if bb coordinates a reasonable
    h, w = image.shape[:2]
    if not ((0 < p1[0] <= w) and (0 < p2[0] <= w) and (0 < p2[1] <= h) and (0 < p2[1] <= h)):
        raise ValueError("BB coordinates `p1` and `p2` violate image width/height and/or are <= 0")

    ## Check image an color
    assert_ndarray_image(image)
    if color is not None:
        color = _colors.convert_color(color, "rgb")
    else:
        _colors.random_color(max_rgb=150)  # Not to bright

    # Draw BB
    _cv2.rectangle(image, p1, p2, color=color, thickness=line_thickness)

    # Draw text and confidence score
    text = ""
    if label: text += label
    if conf:
        if label: text += ": "
        text += str(round(conf * 100, 3)) + "%"
    if text:
        new_p2 = (p1[0] + 10 * len(text), p1[1] - 15)
        _cv2.rectangle(image, p1, new_p2, color=color, thickness=-1)
        _cv2.putText(image, text, (p1[0], p1[1] - 2), _cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, _cv2.LINE_AA)


def load_image(path: str, load_type: str = "unchanged_bgr"):
    """
    Load an image in np.ndarray format. Support grayscale, RGB, BGR, RGBA and BGRA

    @param path: Image path
    @param load_type: Image format that will be return, legal values:
                      ['grey', 'gray', 'rgb', 'bgr', 'unchanged_bgr', 'unchanged_rgb']
    @param return: Image at `path` in the format specified by `load_type`

    """

    # Simple Checks
    _type_check.assert_type(path, str)
    _input_output.assert_path(path)

    # Check load_type
    legal_load_types = {"grey": _cv2.IMREAD_GRAYSCALE,
                        "gray": _cv2.IMREAD_GRAYSCALE,
                        "rgb": _cv2.IMREAD_COLOR,
                        "bgr": _cv2.IMREAD_COLOR,
                        "unchanged_bgr": _cv2.IMREAD_UNCHANGED,
                        "unchanged_rgb": _cv2.IMREAD_UNCHANGED}

    if load_type.lower() not in legal_load_types:
        raise ValueError(f"Expected `load_type` to be in `{list(legal_load_types.keys())}`,"
                         f"but recived `{load_type}`")

    if load_type in ["rgb", "bgr"]:
        if len(_cv2.imread(path, _cv2.IMREAD_UNCHANGED).shape) == 2:
            _warnings.warn(f"`{path}` contains a greyscale image, but recived `load_type = {load_type}`"
                          f"Will return a greyscale image in RGB format i.e. R = G = B")

    # Load image
    image = _cv2.imread(path, legal_load_types[load_type])
    if load_type in ["rgb", "unchanged_rgb"]:
        image = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)

    return image

__all__ = [
    "is_ndarray_greyscale",
    "assert_ndarray_image",
    "pillow_resize_image",
    "pillow_image_to_ndarray",
    "image_size_from_path",
    "get_image_from_url",
    "ndarray_image_to_pillow",
    "ndarray_resize_image",
    "show_image",
    "show_ndarray_image",
    "ndarray_image_center",
    "cv2_cutout_square",
    "cv2_sobel_edge_detection",
    "cv2_draw_bounding_boxes",
    "load_image",
]