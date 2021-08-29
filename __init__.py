# TODO:
# Integrate pandas_profiling in some way, perhaps just a "print what you're supposed to do" kinda thing
# Make pandas print helper: "df.describe(), df.info() ..." just a bunch of different pandas commands in one place
# "jupyter nbconvert readme.ipynb --to markdown"
# Make plot numpy image function

# Sound to mel-spectrogram
# seaborn.set_context(context="talk")  |  https://seaborn.pydata.org/generated/seaborn.set_context.html
# cv2 text to image helper


# Plotly build-in color schemes and other plotly stuff perhaps an entire lib.
# 	px.colors.qualitative.swatches().show()
# 	pio.templates.default = "seaborn"
# 	print(px.colors.qualitative.Plotly)
#________________________________________________________________________________________________________________


# Import those modules which are mean to be visible
from utils._code import (
    all_around,
    colors,
    formatting,
    images,
    imports,
    input_output,
    jupyter,
    pytorch,
    system_info,
    type_check,
)
from utils._code.country_converter import country_converter
