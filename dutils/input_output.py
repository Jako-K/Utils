"""
Description:
Helper functions for working with I/O
"""


import os as _os
import sys as _sys
import pathlib as _pathlib
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt
import json as _json
from glob import glob as _glob
import pickle as _pickle
import shutil as _shutil
import re as _re

from . import type_check as _type_check
from . import system_info as _system_info



def assert_path(path:str):
    """ Assert path exist """
    _type_check.assert_type(path, str)
    if not path_exists(path):
        raise ValueError(f"Received bad path: `{path}`")


def assert_path_dont_exists(path:str):
    """ Assert path don't exist """
    _type_check.assert_type(path, str)
    if path_exists(path):
        raise ValueError(f"Path `{path}` already exists`")


def path_exists(path:str):
    """ Check if path exists or not and return False/True"""
    _type_check.assert_type(path, str)
    return _os.path.exists(path)


def get_system_paths():
    """ Return system paths that are currently available """
    return _sys.path


def add_path_to_system(path:str):
    _type_check.assert_type(path, str)
    assert_path(path)
    _sys.path.append(path)

    if path not in get_system_paths():
        raise AssertionError("This should not be possible!")


def get_file_extension(file_name:str):
    """
    Extract file extension(s) from file_name or path

    Example:
    >> get_file_extension("some_path/works_with_backslashes\\and_2x_extensions.tar.gz")
    '.tar.gz'
    """
    # Checks
    _type_check.assert_type(file_name, str)
    if file_name.find(".") == -1:
        raise ValueError("`file_name` must contain at least 1 `.`, but received 0")

    return ''.join(_pathlib.Path(file_name).suffixes)


def get_current_directory():
    return str(_pathlib.Path().absolute())


def save_plt_plot(save_path:str, fig:_matplotlib.figure.Figure=None, dpi:int=300):
    """ Save `fig` at `save_path` with the quality defined by `dpi`. """
    # Checks
    _type_check.assert_types([save_path, fig, dpi], [str, _matplotlib.figure.Figure, int], [0, 1, 0])
    extension = get_file_extension(save_path)
    if extension not in [".png", ".jpg", ".pdf"]:
        raise ValueError(f"Expected file extension to be in ['png', 'jpg', 'pdf'],"
                         f" but received `{extension}` extension")
    if fig is None:
        _plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


def get_file_basename(path:str, with_extension:bool=False, assert_path_exists:bool=True):
    """
    Function description:
    Extract basename from `path`.

    Example:
    >> get_file_basename('C:/Users/JohnDoe/Desktop/test.png', with_extension=True)
    'test.png'

    @param path: Path basename is supposed to be extracted from
    @param with_extension: Weather or not the file extension e.g. ".png" is to be included
    @param assert_path_exists: if `path` must exists on the current system
    @return: basename with or without file extension
    """

    # Checks
    _type_check.assert_types([path, with_extension, assert_path_exists], [str, bool, bool])
    if assert_path_exists: assert_path(path)
    if with_extension and (path.find(".") == -1):
        raise ValueError("`path` must contain at least  one `.` when `with_extension` is True, but received 0")

    # Extract basename
    basename = _os.path.basename(path)
    if not with_extension and (basename.find(".") != -1):
        basename = basename.split(".")[-2]

    return basename


def write_to_file(file_path:str, append_str:str):
    """
    Adds `append_str` to the end of the file located at `file_path`

    @param file_path: Path to the file. This file must exists already, and will cause an error if not.
    @param append_str: String that's added to the end of the file
    @return:
    """
    # Checks
    _type_check.assert_types([file_path, append_str], [str, str])
    assert_path(file_path)

    file = open(file_path, mode="a")
    print(append_str, file=file, end="")
    file.close()


def read_json(path:str):
    """ Read .json file and return it as string"""
    assert_path(path)
    if path[-5:] != ".json":
        raise ValueError("Expected .json file extension, but received something else")

    f = open(path)
    data = _json.load(f)
    f.close()

    return data


def get_number_of_files(path:str):
    """ Return the total number of files (including folders) in `path`"""
    assert_path(path)
    return len(_glob(_os.path.join(path, "*")))


def read_file(path:str):
    """ Read file at `path` and return it as string"""
    assert_path(path)

    f = open(path, "r")
    return_string = f.read()
    f.close()

    return return_string


def save_as_pickle(obj:object, file_name:str, save_path:str=None):
    """
    Save object as a pickle file in `save_path` with basename `file_name`
    If `save_path` is None, the current working directory is used.
    """
    _type_check.assert_types([file_name, save_path], [str, str], [0, 1])
    if get_file_extension(file_name).find(".pkl") == -1:
        raise ValueError("Expected .pkl file extension, but received something else")

    # Path
    if save_path is None:
        save_path = _os.getcwd()
    else:
        assert_path(save_path)

    full_path = _os.path.join(save_path, file_name)

    with open(full_path, 'wb') as output:
        _pickle.dump(obj, output, _pickle.HIGHEST_PROTOCOL)


def load_pickle_file(path:str):
    """ Load pickle-object and return as is """
    assert_path(path)
    if get_file_extension(path).find(".pkl") == -1:
        raise ValueError("Expected .pkl file extension, but received something else")

    with open(path, 'rb') as pickle_file:
        return _pickle.load(pickle_file)


def copy_folder(from_path:str, to_path:str):
    """ Copy folder from `from_path` to `to_path`. Cannot copy folder if `to_path` already exists"""
    assert_path(from_path)
    assert_path_dont_exists(to_path)
    if not _os.path.isdir(from_path):
        raise ValueError("Expected `from_path` to be a folder, but received something else")

    _shutil.copytree(from_path, to_path)
    assert _os.path.exists(to_path), "Copying folder was unsuccessful, should not be possible"


def is_folder(path: str, must_be_empty: bool = False):
    """
    Check if `path` is a and potentially if it's empty which is determined by `must_be_empty`.

    @param path: A path which is valid on the system which python is running on
    @param must_be_empty: Check if the folder specified by `path` is empty.

    >> is_folder("./path_to_valid_empty_folder", must_be_empty=True)
    True
    """
    _type_check.assert_types([path, must_be_empty], [str, bool])
    is_ok = _os.path.isdir(path)
    if must_be_empty and is_ok:
        is_empty = get_number_of_files(path) == 0
        is_ok = is_ok and is_empty

    return is_ok


def make_file(path: str, allow_override: bool = False):
    """
    Make a new file at the location specified by `path`.
    """

    # Checks
    _type_check.assert_types([path, allow_override], [str, bool])
    if allow_override and is_file(path):
        _os.remove(path)
    elif not allow_override:
        assert_path_dont_exists(path)

    fp = open(path, 'x')
    fp.close()


def is_file(path: str, extension: str = None):
    """
    Check if `path` is a file and if it ends on `extension` if its specified.

    @param path: A path which is valid on the system which python is running on
    @param extension: A file extension which should be present in `path` e.g. `.py`

    # Example
    >> is_file("a/real/path/something/file.py", extension=".py")
    True
    >> is_file("a/real/path/something/file", extension=".py")
    False

    """
    _type_check.assert_types([path, extension], [str, str], [0, 1])
    is_ok = _os.path.isfile(path)
    if extension is not None:
        if extension.find(".") == -1:
            raise ValueError("Expected `extension` to have at least one `.` but received zero.")
        extension_ok = get_file_extension(path) == extension
        is_ok = is_ok and extension_ok

    return is_ok


def remove_file(path: str):
    """
    Remove `path` if it's a valid file.
    """

    assert_path(path)
    if not is_file(path):
        if is_folder(path):
            raise ValueError("`path` is interpreted as a folder, not a file")
        else:
            raise ValueError("`path` is not recognized as a file")
    _os.remove(path)


def get_line_count_file(file_path: str, exclude_empty_lines: bool = True):
    """
    Count number of lines in `file_path`.
    """

    # Checks
    _type_check.assert_types([file_path, exclude_empty_lines], [str, bool], [0, 0])
    if not is_file(file_path):
        raise ValueError("`file_path` is not recognized as a file")

    # Counting
    with open(file_path, 'r') as f:
        if exclude_empty_lines:
            line_count = sum([0 if line.isspace() else 1 for line in f])
        else:
            line_count = sum([1 for _ in f])
    return line_count


def get_line_counts_folder(folder_path: str, only_extension: str = None, exclude_empty_lines: bool = True):
    """
    Calculates the number of lines in a folder.
    NOTE: Ignore every file within `folder_path` which is not recognized as a file.

    @param folder_path: path to the folder
    @param only_extension: only includes file with that extension name
    @param exclude_empty_lines: Don't count empty lines
    @param return: Total number of lines within `folder_path` and
                   a dict of filenames with their corresponding line count.
    """

    # Checks
    _type_check.assert_types(
        [folder_path, only_extension, exclude_empty_lines], [str, str, bool], [0, 1, 0])
    if not is_folder(folder_path):
        raise ValueError(f"Expected `folder_path` to be a folder, but received "
                         f"{'a file' if is_file(folder_path) else 'something else'}")
    if (only_extension is not None) and ("." not in only_extension):
        raise ValueError("Expected `only_extension` to have at least one `.` but received 0")

    line_counts = {}
    glob_path = _os.path.join(folder_path, f"*{only_extension if only_extension else ''}")
    for path in _glob(glob_path):
        if not is_file(path): continue
        base_name = get_file_basename(path, with_extension=True)
        line_counts[base_name] = get_line_count_file(path, exclude_empty_lines)

    return sum(line_counts.values()), line_counts


def get_file_size(path:str, size_prefix:str="m", as_string:bool=True, SI_unit:bool=False):
    """
    Return file size at `path` prefixed by `size_prefix`. `as_string` adds size prefix e.g. megabyte
    NOTE: Windows uses the "binary" prefix for storage. This means 1 Kilobyte is 2^10 bytes, instead of the normal SI Kilo (10^3).

    EXAMPLE:
    >> get_file_size("../my_folder_path/my_file_name.extension", "g", as_string=True)
    '1.185318 gb'

    @param path: Path to a file. NOTE: only files are accepted, passing a folder path will cause an error
    @param size_prefix: Specify the size format. Accepted values: [None, "k", "m", "g"] -> None=Byte, "k"="KiloByte" ..
    @param as_string: Convert the file size to a string and adds the prefix e.g. 1024.0 -> "1.0 kb"
    @param SI_unit: Use SI-units (e.g. kilo=10^3) to prefix bytes if true. If false, the "binary" equivalent is used (e.g. kilo=1024)
    @return: file size in string with units or as a simple float
    """

    # Checks
    _type_check.assert_types([path, as_string, SI_unit], [str, bool, bool])
    _type_check.assert_in(size_prefix, [None, "k", "m", "g"])
    assert_path(path)
    if not is_file(path): raise ValueError("`path` most point to a file")

    # Calculate the actual size of the file
    kilo = 1000 if SI_unit else 1024
    prefix_numeric = {None:1, "k": kilo**1, "m": kilo**2, "g": kilo**3}[size_prefix]
    rounding_amount = {None: 0, "k": 1, "m": 3, "g": 6}[size_prefix]
    file_size = round(_os.path.getsize(path) / prefix_numeric, rounding_amount)

    # The formatting add units according to inputs e.g. file_size = 123000 -> "120.11 kiB"
    # NOTE: as far as i can tell, the consensus around capital letters is: KiB=1024 and kB=1000.
    if SI_unit:
        return f"{file_size} {size_prefix if size_prefix else ''}B" if as_string else file_size
    else:
        return f"{file_size} {size_prefix.upper() if size_prefix else ''}iB" if as_string else file_size

# TODO Unit tests
def search_file(file_path: str, search_query: str, lazy_search: bool = False, regex_flag: _re.RegexFlag = 0) -> dict:
    """
    Search `file_path` for regex expression `search_query` and
    return all occurrences with their corresponding line number.
    if `lazy_search` is True, only the first occurrence is returned.
    NOTE: No checks is performed to ensure unicode compatible. This
          means that "image.png" will throw a UnicodeDecodeError

    EXAMPLE:
    >> search_file(file_path, "something", True, regex_flag=re.IGNORECASE)
    {6: 'something'}

    @param file_path: path to file (this cannot be a folder)
    @param search_query: regex expression used for search in `file_path`
    @param lazy_search: If true, only the first occurrence is returned.
    @param regex_flag: Add functionality e.g. `re.IGNORECASE` (has enum value 2)
    @return: dict with line numbers as keys and string matches as values
    """

    # Checks
    _type_check.assert_types([file_path, search_query, lazy_search, regex_flag], [str, str, bool, int])
    if not is_file(file_path):
        raise ValueError(f"`{file_path}` is not a file. Did you perhaps pass a folder?")
    if search_query == "":
        raise ValueError("Received an empty `search_query`.")

    # Setup
    file_str = read_file(file_path)
    findings = {}

    # Stop after finding the first occurrences of `search_query`
    if lazy_search:
        matched = _re.search(search_query, file_str, flags=regex_flag)
        if not matched: return findings
        start, end = matched.span()
        line_number = len(_re.findall(r"\n", file_str[:start])) + 1
        findings[line_number] = file_str[start: end]
        return findings

    # Find all occurrences of `search_query`findings = {}
    for match in _re.finditer(search_query, file_str, flags=regex_flag):
        if not match: break
        start, end = match.span()
        line_number = len(_re.findall(r"\n", file_str[:start])) + 1
        findings[line_number] = file_str[start: end]

    return findings

# TODO Unit tests
def search_folder(folder_path: str, search_query: str, lazy_search: bool = True, regex_flag: _re.RegexFlag = 0):
    """
    Search all readable files in `folder_path` for regex expression `search_query` and
    return all occurrences for each file. if `lazy_search` is True, only the first occurrence
    in each file is returned.

    # EXAMPLE
    >> U.experimental.search_folder("./yolov3/", "Mosaic")
    {'./yolov3/detect.py': {43: 'IOU'}}

    @param folder_path: path to folder
    @param search_query: regex expression used for search in `folder_path`
    @param lazy_search: If true, only the first occurrence for is file is returned.
    @param regex_flag: Add functionality e.g. `re.IGNORECASE` (has enum value 2)
    @return: dict with line numbers as keys and string matches as values
    """

    # Checks
    _type_check.assert_types([folder_path, search_query, lazy_search, regex_flag], [str, str, bool, int])
    if not is_folder(folder_path):
        raise ValueError("`folder_path` is not a folder. Did you perhaps pass a file?")
    if search_query == "":
        raise ValueError("Received an empty `search_query`")

    # Recursively extract all paths in the folder
    file_paths = _glob(_os.path.join(folder_path, "**", "*.*"), recursive=True)
    if _system_info.on_windows(): file_paths = [p.replace("\\", "/") for p in file_paths]  # Avoid Windows' backslash in paths

    # Seems like the most reasonable thing to do if there's no files in the received folder
    if not any([is_file(p) for p in file_paths]):
        raise ValueError("`folder_path` contains 0 files")

    # Search each readable file for `search_query`
    findings = {}
    exclude_extensions = [".exe"] + audio_formats + compression_formats + video_formats + image_formats
    for path in file_paths:
        # try and exclude non-unicode files (e.g. ".MP4"), but wrap in a try/except to be sure
        if get_file_extension(path).lower() in exclude_extensions: continue
        try:
            findings[path] = search_file(path, search_query, lazy_search, regex_flag)
        except UnicodeDecodeError:
            pass

    # Remove all paths without any matches, convert to absolute paths, replace backslashes and return
    if _system_info.on_windows():
        return {_os.path.abspath(k).replace("\\", "/"): v for (k, v) in findings.items() if v}
    return {_os.path.abspath(k): v for (k, v) in findings.items() if v}




image_formats = ['.png', '.jpg', '.jpeg', '.jpeg2000', '.tiff', '.bmp', '.gif', '.raw', '.psd', '.cr2']
video_formats = ['.mp4', '.mov', '.avi', '.flv', '.mkv', '.wmv', '.avchd', '.webm', '.h.264', '.mpeg-4']
audio_formats = ['.3ga', '.aac', '.aiff', '.amr', '.ape', '.arf', '.asf', '.asx', '.cda', '.dvf', '.flac', '.gp4', '.gp5',
                 '.gpx', '.logic', '.m4a', '.m4b', '.m4p', '.midi', '.mp3', '.ogg', '.opus', '.pcm', '.rec', '.snd', '.sng',
                 '.uax', '.wav', '.wma', '.wpl', '.zab']
compression_formats = ['.7z', '.arj', '.bin', '.bzip', '.bzip2', '.cab', '.cpio', '.deb', '.eargz', '.hqx', '.jar', '.lha',
                       '.rar', '.rpm', '.sea', '.sit', '.tar', '.war', '.zip']


__all__ = [
    "assert_path",
    "assert_path_dont_exists",
    "path_exists",
    "get_system_paths",
    "add_path_to_system",
    "get_file_extension",
    "get_current_directory",
    "save_plt_plot",
    "get_file_basename",
    "write_to_file",
    "read_json",
    "get_number_of_files",
    "read_file",
    "save_as_pickle",
    "load_pickle_file",
    "copy_folder",
    "is_folder",
    "make_file",
    "is_file",
    "remove_file",
    "get_line_count_file",
    "get_line_counts_folder",
    "get_file_size",
    "image_formats",
    "video_formats",
    "search_folder",
    "audio_formats",
    "compression_formats",
    "search_file"
]


