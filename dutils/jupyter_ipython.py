# TODO add unit test

import IPython as _IPython
import cv2 as _cv2
import matplotlib.pyplot as _plt
from PIL import Image as _Image
import numpy as _np
import requests as _requests
import validators as _validators
from rectpack import newPacker as _newPacker
import warnings as _warnings
import torch as _torch

from . import type_check as _type_check
from . import input_output as _input_output


def in_jupyter():
    """
    Check if python is currently running in a jupyter enviroment.
    NOTE: The implementation is somewhat dubious, but have tested it to the best of my abilities
    """

    try:
        shell = get_ipython().__class__.__name__ # This is supposed to be an unresolved reference anywhere outside jupyter
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?), going to assume that we are not in jupyter
    except NameError:
        return False  # Probably standard Python interpreter


def assert_in_jupyter():
    if not in_jupyter():
        raise RuntimeError("Failed to detect Jupyter environment")


def adjust_screen_width(percentage:int=75):
    """
    Adjust the cell width of Jupyter notebook. 100 % is the entire browser window.

    @param percentage: Percentage cells are scaled by (min: 50, max: 100)
    @return: None
    """
    # Checks
    _type_check.assert_type(percentage, int)
    assert_in_jupyter()
    if not (50<=percentage<=100):
        raise ValueError(f"Expected 50 <= `percentage` <= 100, but received `{percentage}`")

    # Adjust width
    argument = "<style>.container { width:" + str(percentage) + "% !important; }</style>"
    _IPython.core.display.display(_IPython.core.display.HTML(argument))


def play_audio(path:str, plot:bool=True):
    """
    Load sound from `path` and display it if `plot` is True.
    NOTE: Uses torchaudio as backend, and therefore only supports what they do.

    @param path: Path where sound can be found
    @param plot: Determine if the raw waveform is plotted alongside the playable audio
    @return: None
    """

    import torchaudio as _torchaudio # TODO Move this together with the other imports, when the "sox warning shenanigans" has been fixed

    #Tests
    assert_in_jupyter()
    _input_output.assert_path(path)
    _type_check.assert_type(plot, bool)

    # Audio load and play
    sound, sample_rate = _torchaudio.load(path)
    audio_bar = _IPython.display.Audio(path)
    _IPython.core.display.display(audio_bar)

    if plot:
        duration = round(len(sound[0]) / sample_rate, 3)
        _plt.plot(sound[0])
        _plt.title(f"type: {audio_bar.mimetype} | duration: {duration} s | sample rate: {sample_rate}")


def _get_collage_image(images:list, allow_rotations:bool=False):
    """
    Used to pack `images` into a single image.
    NOTE: This function is only intended to be used by `show_image()`

    @param images: list of images in np.ndarray format
    @param allow_rotations: Determine if the packing algorithm is allowed to rotate the images
    @return: A single collage image build from `images` in `np.ndarray` format
    """

    # A lot of the complexity is removed if all the images are of the same size. Which means a simpler method can be used
    if all([images[0].shape == image.shape for image in images]):
        cols, rows, resize_factor, _ = _get_grid_parameters(images)
        return _get_grid_image(images, cols, rows, resize_factor)

    # Setup
    rectangles = [(s.shape[0], s.shape[1], i) for i, s in enumerate(images)]
    height_domain = [60 * i for i in range(1, 8)] + [int(1080 * 5 / 1.05 ** i) for i in range(50, 0, -1)] # Just different sizes to try
    width_domain = [100 * i for i in range(1, 8)] + [int(1920 * 5 / 1.05 ** i) for i in range(50, 0, -1)]
    canvas_dims = list(zip(height_domain, width_domain))
    canvas_image = None

    # Attempt to pack all images within the smallest of the predefined width-height combinations
    for canvas_dim in canvas_dims:

        # Try packing
        packer = _newPacker(rotation=allow_rotations)
        for r in rectangles: packer.add_rect(*r)
        packer.add_bin(*canvas_dim)
        packer.pack()

        # If all images couldn't fit, try with a larger image
        if len(packer.rect_list()) != len(images):
            continue

        # Setup
        canvas_image = _np.zeros((canvas_dim[0], canvas_dim[1], 3)).astype(_np.uint8) + 64 # 64 seems to be a pretty versatile grey color
        H = canvas_image.shape[0]


        for rect in packer[0]:
            image = images[rect.rid]
            h, w, y, x = rect.width, rect.height, rect.x, rect.y

            # Transform origin to upper left corner
            y = H - y - h

            # Handle image rotations if necessary
            if image.shape[:-1] != (h, w): image = image.transpose(1, 0, 2)
            canvas_image[y:y + h, x:x + w, :] = image
        break

    if canvas_image is None:
        raise RuntimeError("Failed to produce mosaic image. The cause is most likely to many and/or to large images")

    return canvas_image


def _get_grid_parameters(images, max_height=1080, max_width=1920, desired_ratio=9/17):
    """
    Try at estimate #cols, #rows and resizing factor necessary for displaying a list of images in a visually pleasing way.
    This is essentially done by minimizing 3 separate parameters:
        (1) difference between `desired_ratio` and height/width of the final image
        (2) Amount of image scaling necessary
        (3) the number of empty cells (e.g. 3 images on a 2x2 --> empty_cell = 1)

    NOTE1: This was pretty challenging function to write and the solution probably suffered from this.
           I've included some notes at "doc/_get_grid_parameters.jpg" which will hopefully motivate the solution - in
           particular the loss-function used for optimization.
    NOTE2: This function is only intended to be used by `show_image()`

    @param images: list np.ndarray images
    @param max_height:
    @param max_width:
    @param desired_ratio:
    @return: cols, rows, scaling_factor, loss_info
    """

    N = len(images)
    h, w, _ = images[0].shape
    H, W = max_height, max_width

    losses = {}
    losses_split = []
    for a in [0.05 + 0.01 * i for i in range(96)]:
        for x in range(1, N + 1):
            for y in range(1, N + 1):

                # If the solution is not valid continue
                if (h * a * y > H) or (w * a * x > W) or (x * y < N):
                    continue
                # Otherwise calculate loss
                else:
                    ratio_loss = abs((h * y) / (w * x) - desired_ratio) # (1)
                    scale_loss = (1 - a) ** 2 # (2)
                    empty_cell_loss = x*y/N - 1 # (3)
                    losses[(y, x, a)] = ratio_loss + scale_loss + empty_cell_loss
                    losses_split.append([ratio_loss, scale_loss, empty_cell_loss])

    # pick parameters with the lowest loss
    rl, sl, ecl = losses_split[_np.argmin(list(losses.values()))]
    loss_info = {"ratio":rl, "scale":sl, "empty_cell":ecl, "total":rl+sl+ecl}
    cols, rows, scaling_factor = min(losses, key=losses.get)
    return cols, rows, scaling_factor, loss_info


def _get_grid_image(images:list, cols:int, rows:int, resize_factor:float):
    """
    Put a list of np.ndarray images into a single combined image.
    NOTE: This function is only intended to be used by `show_image()`

    @param images: list np.ndarray images
    @param cols: Number of columns with images
    @param rows: Number of rows with images
    @param resize_factor: Image resize factor
    @return: On single picture with all the images in `images`
    """

    h_old, w_old, _ = images[0].shape
    h = int(h_old * resize_factor)
    w = int(w_old * resize_factor)
    canvas = _np.zeros((int(h_old * resize_factor * cols), int(w_old * resize_factor * rows), 3))
    canvas = canvas.astype(_np.uint8)

    image_index = -1
    for row in range(rows):
        for col in range(cols):
            image_index += 1
            if image_index >= len(images): break

            # Load and rescale image
            image = images[image_index]
            image = _cv2.resize(image, (w, h), interpolation=_cv2.INTER_AREA)

            # Add image to the final image
            canvas[col * h: (col + 1) * h, row * w: (row + 1) * w, :] = image
    return canvas


def _get_image(source, resize_factor: float = 1.0, BGR2RGB: bool = None):
    """
    Take an image in format: path, url, ndarray or tensor.
    Returns `source` as np.ndarray image after some processing e.g. remove alpha

    NOTE: This function is only intended to be used by `show_image()`
    """

    # `source` and `resize` checks
    is_path = _input_output.path_exists(source) if isinstance(source, str) else False
    is_url = True if isinstance(source, str) and _validators.url(source) is True else False
    is_ndarray = True if isinstance(source, _np.ndarray) else False
    is_torch_tensor = True if isinstance(source, _torch.Tensor) else False

    if not any([is_path, is_url, is_ndarray, is_torch_tensor]):
        raise ValueError("`source` could not be interpreted as a path, url, ndarray or tensor.")

    if is_path and is_url:
        raise AssertionError("This should not be possible")  # Don't see how a `source` can be a path and a url simultaneously

    if resize_factor < 0:
        raise ValueError(f"`resize_factor` > 0, received value of {resize_factor}")

    if is_torch_tensor and len(source.shape) > 3:
        raise ValueError(f"Expected tensor image to be of shape (channels, height, width), but received: {source.shape}. "
                         f"If you passed a single image as a batch use <YOUR_IMAGE>.squeeze(0). "
                         f"Otherwise pick a single image or split the batch into individual images an pass them all")

    if is_torch_tensor and (len(source.shape) == 3) and (source.shape[0] >= source.shape[1] or source.shape[0] >= source.shape[2]):
        raise ValueError(f"Expect tensor image to be of shape (channels, height, width), but received: {source.shape}. "
                         f"If your image is of shape (height, width, channels) use `<YOUR_IMAGE>.permute(2, 0, 1)`")

    # Cast to Pillow image
    if is_path:
        image = _Image.open(source)
    elif is_url:
        image = _Image.open(_requests.get(source, stream=True).raw)
    elif is_ndarray:
        image = _Image.fromarray(source)
    elif is_torch_tensor:
        corrected = source.permute(1, 2, 0) if (len(source.shape) > 2) else source
        image = _Image.fromarray(corrected.numpy())

    # Swap blue and red color channel stuff
    num_channels = len(image.getbands())
    bgr2rgb_auto = (BGR2RGB is None) and is_ndarray and (num_channels in [3, 4])
    if BGR2RGB or bgr2rgb_auto:
        # BGR --> RGB or BGRA --> RGBA
        as_array = _np.asarray(image)
        color_corrected = _cv2.cvtColor(as_array, _cv2.COLOR_BGR2RGB) if (num_channels == 3) \
            else _cv2.cvtColor(as_array, _cv2.COLOR_BGRA2RGBA)
        image = _Image.fromarray(color_corrected)

    if resize_factor != 1.0:
        width = int(image.size[0] * resize_factor)
        height = int(image.size[1] * resize_factor)
        image = image.resize((width, height), resample=0, box=None)

    image = _np.asarray(image)

    # Adds 3 identical channels to greyscale images (for compatibility)
    if (len(image.shape) == 2) or (image.shape[-1] == 1):
        image = _cv2.cvtColor(image, _cv2.COLOR_GRAY2RGB)

    # Remove alpha channel (for compatibility)
    if image.shape[-1] == 4:
        image = image[:,:,:3]

    return image


def show_image(source, resize_factor:float=1.0, BGR2RGB:bool=None, return_image:bool=False):
    """
    Display a single image or a list of images from path, np.ndarray, torch.Tensor or url.

    @param source: path, np.ndarray, torch.Tensor url pointing to the image you wish to display
    @param resize_factor: Rescale factor in percentage (i.e. 0-1)
    @param BGR2RGB: Convert `source` from BGR to RGB. If `None`, will convert np.ndarray images automatically
    @param return_image: return image as `np.ndarray`
    """

    # Checks
    _type_check.assert_in(type(source), [_np.ndarray, _torch.Tensor, str, list, tuple])
    _type_check.assert_types([resize_factor, BGR2RGB], [float, bool], [0, 1])
    assert_in_jupyter()


    # Prepare the final image(s)
    if type(source) not in [list, tuple]:
        final_image = _get_image(source, resize_factor, BGR2RGB)
    else:
        if len(source) > 200: # Anything above 200 indexes seems unnecessary
            _warnings.warn(
                f"Received `{len(source)}` images, the maximum limit is 200. "
                "Will pick 200 random images from `source` for display and discard the rest"
            )
            random_indexes_200 = _np.random.choice(_np.arange(len(source)), 200)
            source = [source[i] for i in random_indexes_200]

        images = [_get_image(image, resize_factor, BGR2RGB) for image in source]
        final_image = _get_collage_image(images, allow_rotations=False)


    # Resize the final image if it's larger than 2160x3840
    scale_factor = None
    if final_image.shape[1] > 3840:
        scale_factor = 3840/final_image.shape[1]
    elif final_image.shape[0] > 2160:
        scale_factor = 2160 / final_image.shape[0]
    if scale_factor:
        height = int(final_image.shape[0]*scale_factor)
        width = int(final_image.shape[1]*scale_factor)
        final_image = _cv2.resize(final_image, (height, width))

    # Display and return image
    final_image = _Image.fromarray(final_image)
    display(final_image)
    return return_image if return_image else None


def clear_variables():
    assert_in_jupyter()
    get_ipython().magic('reset -sf')


__all__=[
    "in_jupyter",
    "assert_in_jupyter",
    "adjust_screen_width",
    "play_audio",
    "show_image",
    "clear_variables"
]


