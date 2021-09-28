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
#    This is, admittedly, not a pretty solution, but I feel like one should be able to use autocomplete to quickly
#    see which functionality is avaliable. And this would be pretty darn hard if you had to scroll through 10 non-relavent 
#    import such as np, os, sys etc.
#_______________________________________________________________________________________________________________________


# Import modules
_all_modules_str = ["all_around", "colors", "experimental", "formatting", "images", "imports", "input_output",
                    "jupyter_ipython", "pytorch", "system_info", "time_and_date", "type_check", "country_converter"]


# Import all the modules correctly and exposes everything callable to the search function.
all_searchable = []
for module in _all_modules_str:
    exec(f"from . import {module}")
    exec(f"module_all = {module}.__all__") # Every module in `.` has everything which should be visible in `__all__`
    exec("all_searchable += [f'{module}.{s}' for s in module_all]")


def search(name:str):
    """
    Search `dutils` for everything importable which contains `name`.
    NOTE: The function is case insensitive e.g. 'RGB' and 'rgb' are interpreted the same
    """
    if not isinstance(name, str):
        TypeError(f"Expected type `str`, but received type `{type(name)}`")

    matching_results = [search_result for search_result in all_searchable if name.lower() in search_result.lower()]
    return sorted(matching_results)

#________________________________________________________________________________________________________________
# TODO:

# 1.) Fix system info print, when not nvidia and remove pytorch cuda dependencies.
# 2.) Find a way to check that __all__ contains everything
# 3.) Check if there's a raise in front of all errors 
# 4.) Fix the folder structure, such that every sub package e.g. `dutils.image` has its own folder.
#     This would alleviate much of the import shenanigans
# 5.) After the folder structure has been fixed, remove "_names" from __all__ e.g. "colors._assert_color_scheme"
#     which is only there for testing purposes
# 6.) pytorch: 
#        * Save and load model.
#        * Add "on_kaggle" option to U.pytorch.templates.config_file, such that it uses Kaggle's secrets to log in to WB
# 



# NOTES AND IDEAS:

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