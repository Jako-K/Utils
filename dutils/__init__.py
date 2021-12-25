"""
DESCRIPTION:
A bunch of modules with helper functions and alike.
NOTE: `dutils.search("what_you_want_to_search_for")` can help you find what you are looking for

EXAMPLE:
>> `dutils.search("image")`
['images.get_image_from_url', 'images.show_image', ...]
"""


from glob import glob as _glob
import os as _os

# Prepare modules for search function and for import
_folder_path_for_glob = _os.path.join(_os.path.dirname(__file__), "*")
_all_file_names = [_os.path.basename(path) for path in _glob(_folder_path_for_glob)]
_all_modules_str = []
for _name in _all_file_names:
    if (_name[-3:] == ".py") and (_name[0] != "_"):
        _all_modules_str.append(_name[:-3])


# Import all the modules correctly and exposes everything callable to the search function.
_all_searchable = []
for _module in _all_modules_str:
    exec(f"from . import {_module}")
    exec(f"_module_all = {_module}.__all__") # Every module in `.` has everything which should be visible in `__all__`
    exec("_all_searchable += [f'{_module}.{s}' for s in _module_all]")


def search(name:str):
    """
    Search `dutils` for everything importable which contains `name`.
    NOTE: The function is case insensitive e.g. 'RGB' and 'rgb' are interpreted the same
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected type `str`, but received type `{type(name)}`")

    matching_results = [search_result for search_result in _all_searchable if name.lower() in search_result.lower()]
    return sorted(matching_results)


# TODO:
#_______________________________________________________________________________________________________________________

# system_info
#   * Fix print when not nvidia and remove pytorch cuda dependencies.

# Pytorch:
#   * Save and load model.
#   * Refactor and delete unused functions
#   * Add missing checks
#   * Custom DataLoader that can .to(device) and do some custom stuff before getting batch e.g. squeeze(1)
#   * Need a way to test the templates
#   * template: dataset/dataloader setup with train/valid split
#   * template: optimizer = C.optimizer(model.parameters(), **C.optimizer_hyper)
#               scheduler = C.scheduler(optimizer, **C.scheduler_hyper)
#               criterion = C.criterion
#   * template: cross validation loop?
#   * template: add a safe model per epoch plot?
#   * add a function: bucket continuous data for stratified folds

# Color:
#    * add HSV and BGR support
#    * Add @param to all
#    * I've heard that random color creation is much better in HSV (less muddy colors) and then convert to RGB. Try it

# Misc:
#   * Sound to mel spectrogram
#   * cv2 text to image helper
#   * Integrate pandas_profiling in some way, perhaps just a "print what you're supposed to do" kinda thing

# Images:
#   * `show_ndarray_image` take list of images for grid image display
#   * Image combine function, with an adjustable merge parameter i.e. what percentage `p` of image_a and image_b=(1-p)
#   * to_greyscale

# input_output:
#   * @param to all functions

# type_checks:
#   * Add a is_list_like function: which would for instance accept both: list, tuples, np.ndarray etc.
#     but not something like a string even though it has __iter__
#   * This is confusing: assert_comparison_number(rows, 1, ">=", "rows") change to order to: (rows, ">=", 1, "rows")


# Unit tests:
#   * Images
#   * Pytorch
#   * Colors
#   * Jupyter: Would it be possible to run the test_jupyter.ipynb notebook from within the test_all.py file?
# ______________________________________________________________________________________________________________________



# NOTES:
#_______________________________________________________________________________________________________________________
# *  The whole "underscore in front of modules" shenanigans I've used in most of the modules is only there to prevent
#    namespace pollution e.g. `dutils.images.np` is avoided this way and `np` will therefore not be visible to the user
#    This is, admittedly, not a pretty solution, but I feel like you should be able to use autocomplete to quickly
#    browse which functionality is available. And this would be pretty darn hard if you had to scroll through a bunch
#    of irrelevant imports such as np, os, sys etc.
#_______________________________________________________________________________________________________________________



# NOTES, IDEAS AND RANDOM THOUGHTS:
# ______________________________________________________________________________________________________________________

# The image module could have a Image class which could hold stuff like format e.g. "RGB"
# Should country_converter even be here? There must be some other implementation which has already done it?
# seaborn.set_context(context="talk")  |  https://seaborn.pydata.org/generated/seaborn.set_context.html
# Add color print option e.g. print_in(text, "orange")
# Plotly build-in color schemes and other plotly stuff perhaps an entire module.
# 	px.colors.qualitative.swatches().show()
# 	pio.templates.default = "seaborn"
# 	print(px.colors.qualitative.Plotly)
# ______________________________________________________________________________________________________________________