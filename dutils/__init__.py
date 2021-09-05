"""
DESCRIPTION:
A bunch modules with helper functions.
`dutils.search("what_you_want_to_search_for")` can help you find what you're looking for

EXAMPLE:
>> `dutils.search("image")`
['images.get_image_from_url', 'images.show_image', ...]
"""

# NOTES:
#_______________________________________________________________________________________________________________________
# *  The whole "underscore in front of modules" shenanigans I've used in most of the modules is only there to prevent
#    namespace pollution e.g. `dutils.images.np` is avoided this way and `np` will therefore not be visible to the user
#_______________________________________________________________________________________________________________________

# I want to avoid `from dutils import dutils` shenanigans, so everything should be accessible like `dutils.<module_name>`
# To make that happened, I need a way of loading all modules within `./_code`. This was the only solution I could
# find that didn't require a total folder restructuring (Its probably not ideal, but it gets the job done).
from os.path import basename as _basename
from glob import glob as _glob
import sys as _sys
import os as _os
_folder_path = _os.path.abspath(__file__)[:-12]
_sys.path.append(_os.path.join(_folder_path, "_code"))

# Find and import all the `.py` files within `_code` which is not prefixed by an underscore
_modules_names = [_basename(file)[:-3] for file in
                 _glob(_os.path.join(_folder_path, '_code/*.py')) if _basename(file)[0] != "_"]

for module_name in _modules_names:
    exec(f'{module_name} = __import__("{module_name}")')

# Import certain things from those modules which should only be partially visible
from ._code.country_converter import country_converter
from ._code._search import search


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