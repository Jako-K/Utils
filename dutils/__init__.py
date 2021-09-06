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

from ._code import (
    all_around,
    colors,
    experimental,
    formatting,
    images,
    imports,
    input_output,
    jupyter_ipython,
    pytorch,
    system_info,
    time_and_date,
    type_check
)

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