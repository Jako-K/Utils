import type_check as _type_check
from os.path import basename as _basename
from glob import glob as _glob

def search(name:str):
    """
    DESCRIPTION:
    Search `utils` for everything importable which contains `name`.

    NOTE: The function is case insensitive e.g. 'RGB' and 'rgb' are interpreted the same
    """
    _type_check.assert_type(name, str)
    modules_names = [_basename(file)[:-3] for file in _glob("./_code/*.py") if _basename(file)[0] != "_"]
    all_searchable = []
    for module in modules_names:
        all_searchable += [f"{module}.{s}" for s in __import__(module).__all__]

    return [search_result for search_result in all_searchable
            if name.lower() in search_result.lower()]

__all__ = ["search"]
