"""
A bunch of helper functions.

`utils.name_search` can help you find what you're looking for
Example:
`utils.name_search("image")` will provide a list of everything with `image` in it e.g. `get_image_from_url`
"""

# NOTES:
#_______________________________________________________________________________________________________________________
# *  The whole "underscore in front of modules" shenanigans I've used throughout the package is only there to prevent
#    namespace pollution e.g. `utils.images.np` is avoided this way and `np` will not be visible to the user
#_______________________________________________________________________________________________________________________

# I want to avoid `from utils import utils` shenanigans, so everything should be accessible like `utils.<module_name>`
# And to make that happened I need a way of loading all modules within `./_code`. This was the only solution I could
# find which didn't required a total folder restructuring (Its probably not ideal, but it gets the job done).
from os.path import basename as _basename
from glob import glob as _glob
import sys as _sys
_sys.path.append("./_code")

# Find and import all the `.py` files within `_code` which is not prefixed by an underscore
modules_names = [_basename(file)[:-3] for file in _glob("./_code/*.py") if _basename(file)[0] != "_"]
for module_name in modules_names:
    exec(f'{module_name} = __import__("{module_name}")')


# Import certain things from those modules which should only be partially visible
from utils._code.country_converter import country_converter
from utils._code._name_search import name_search








#________________________________________________________________________________________________________________
# TODO:
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