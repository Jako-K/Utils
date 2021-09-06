from . import type_check as _type_check
from os.path import basename as _basename
from glob import glob as _glob
import os as _os
import sys as _sys
_code_path = _os.path.abspath(__file__)[:-11]
_sys.path.append(_code_path)

def search(name:str):
    """
    DESCRIPTION:
    Search `utils` for everything importable which contains `name`.
    NOTE: The function is case insensitive e.g. 'RGB' and 'rgb' are interpreted the same
    """
    _type_check.assert_type(name, str)
    glob_path = _os.path.join(_code_path, "*.py")
    modules_names = [_basename(file)[:-3] for file in _glob(glob_path) if _basename(file)[0] != "_"]
    all_searchable = []

    print(modules_names)

    import importlib
    for module in modules_names:
        all_searchable += [f"{module}.{s}" for s in __import__(module).__all__]

    return [search_result for search_result in all_searchable
            if name.lower() in search_result.lower()]

__all__ = ["search"]
