"""
DESCRIPTION:
A bunch of modules with task specific helper functions and alike.
NOTE: `dutils.search("what_you_want_to_search_for")` can help you find what you're looking for

EXAMPLE:
>> `dutils.search("image")`
['images.get_image_from_url', 'images.show_image', ...]
"""

# NOTES:
#_______________________________________________________________________________________________________________________
# *  The whole "underscore in front of modules" shenanigans I've used in most of the modules is only there to prevent
#    namespace pollution e.g. `dutils.images.np` is avoided this way and `np` will therefore not be visible to the user
#    This is, admittedly, not a pretty solution, but I feel like you should be able to use autocomplete to quickly
#    browse which functionality is avaliable. And this would be pretty darn hard if you had to scroll through a bunch
#    of irrelevant imports such as np, os, sys etc.
#_______________________________________________________________________________________________________________________


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

#________________________________________________________________________________________________________________
# TODO:

# 1.) Fix system info print, when not nvidia and remove pytorch cuda dependencies.
# 2.) pytorch: 
#    * Save and load model.
# 3.) Add unit tests to images and to pytorch
# 



# NOTES, IDEAS AND RANDOM THROUGHTS:

# Integrate pandas_profiling in some way, perhaps just a "print what you're supposed to do" kinda thing
# Make pandas print helper: "df.describe(), df.info() ..." just a bunch of different pandas commands in one place
# "jupyter nbconvert readme.ipynb --to markdown"
# Make plot numpy image function

# Sound to mel-spectrogram
# seaborn.set_context(context="talk")  |  https://seaborn.pydata.org/generated/seaborn.set_context.html
# cv2 text to image helper

# Add color print option e.g. print_in(text, "orange")

# Plotly build-in color schemes and other plotly stuff perhaps an entire module.
# 	px.colors.qualitative.swatches().show()
# 	pio.templates.default = "seaborn"
# 	print(px.colors.qualitative.Plotly)
#________________________________________________________________________________________________________________