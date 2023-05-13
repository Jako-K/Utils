import inspect as _inspect
import os as _os
import types as _types
import pathlib as _pathlib

from . import type_check as _type_check
from . import jupyter_ipython as _jupyter

def get_imports(all_requests:list=None):
    """
    Return common imports e.g. import matplotlib.pyplot as plt, import pandas as pd etc.
    Expect `all_requests` to be in `["torch", "torchvision", "all_around", "all"]`
    and to be a list e.g. ["all"]
    """
    if all_requests is None: all_requests = ["all_around"] # To avoid mutable default argument
    legal_imports = ["torch", "torchvision", "all_around", "all"]

    # Checks
    _type_check.assert_type(all_requests, list)
    for request in all_requests:
        if request not in legal_imports:
            raise ValueError(f"Received bad request {request}. " f"Accepted requests are: {legal_imports}")

    torch_imp = \
    """
    # Torch
    import wandb
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split\
    """

    vision_imp = \
    """
    # Torch vision stuff
    import cv2
    import torchvision
    import albumentations as album
    import albumentations.pytorch\
    """

    all_around_imp = \
    f"""
    # All around
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (20, 6)
    import seaborn as sns; sns.set_style("whitegrid")
    {"from tqdm.notebook import tqdm" if _jupyter.in_jupyter() else "from tqdm import tqdm"}
    import pandas as pd
    import numpy as np
    from glob import glob
    import sys
    import os\
    """

    # Perform import print statements
    to_concate = None
    if "all" in all_requests:
        to_concate = [torch_imp, vision_imp, all_around_imp]
    else:
        for request in all_requests:
            to_concate = []
            if request == "torchvision":
                to_concate = [torch_imp, vision_imp]
            elif (request == "torch") and (sum([r == "torchvision" for r in all_requests]) == 0): # Avoid double torch import
                to_concate = [torch_imp]
            elif request == "all_around":
                to_concate = [all_around_imp]

    for all_imports in to_concate:
        [print(line[4:]) for line in all_imports.split("\n")] # <-- line[4:] remove 4 start spaces


def get_available_functions(module:_types.ModuleType):
    """ Return all public functions in `module` """
    _type_check.assert_type(module, _types.ModuleType)
    return [func for func, _ in _inspect.getmembers(module, _inspect.isfunction)]


def get_all_available_import_classes(module:_types.ModuleType):
    """ Return all public classes in `module` """
    _type_check.assert_type(module, _types.ModuleType)
    return [func for func, _ in _inspect.getmembers(module, _inspect.isclass)]


def get_module_path(module:_types.ModuleType):
    """ Return absolute path of `module` """
    _type_check.assert_type(module, _types.ModuleType)
    return str(_pathlib.Path(module.__file__).resolve())


__all__=[
    "get_imports",
    "get_available_functions",
    "get_all_available_import_classes",
    "get_module_path"
]



