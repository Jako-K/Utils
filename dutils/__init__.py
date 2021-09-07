"""
DESCRIPTION:
A bunch modules with helper functions.
NOTE: `dutils.search("what_you_want_to_search_for")` can help you find what you're looking for

EXAMPLE:
>> `dutils.search("image")`
['images.get_image_from_url', 'images.show_image', ...]
"""

# NOTES:
#_______________________________________________________________________________________________________________________
# *  The whole "underscore in front of modules" shenanigans I've used in most of the modules is only there to prevent
#    namespace pollution e.g. `dutils.images.np` is avoided this way and `np` will therefore not be visible to the user
#_______________________________________________________________________________________________________________________


# Import main modules
_all_modules_str = ["all_around", "colors", "experimental", "formatting", "images", "imports", "input_output",
                    "jupyter_ipython", "pytorch", "system_info", "time_and_date", "type_check", "country_converter"]

# This is admittedly not a pretty solution, but believe or not, this was the only way i could get dynamic imports
# and a search function working without all sorts shenanigans. Just move on, don't worry about it :)
all_searchable = []
for module in _all_modules_str:
    exec(f"from ._code import {module}")
    exec(f"module_all = {module}.__all__") # Every module in `_code` has everything which should be visible in `__all__`
    exec("all_searchable += [f'{module}.{s}' for s in module_all]")

import os as _os
_code_path = _os.path.abspath(__file__)[:-11]

def search(name:str):
    """
    DESCRIPTION:
    Search `utils` for everything importable which contains `name`.
    NOTE: The function is case insensitive e.g. 'RGB' and 'rgb' are interpreted the same
    """
    if not isinstance(name, str):
        TypeError(f"Expected type `str`, but received type `{type(name)}`")

    matching_results = [search_result for search_result in all_searchable if name.lower() in search_result.lower()]
    return sorted(matching_results)

#________________________________________________________________________________________________________________
# TODO and ideas:

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
#
#________________________________________________________________________________________________________________