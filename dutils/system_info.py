"""
DESCRIPTION:
Provide functionality to get information about hardware and software (including operation system)
"""

import os as _os
import subprocess as _subprocess
import platform as _platform
from tkinter import Tk as _Tk
import torch as _torch


windows_illegal_file_name_character = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]


def get_vram_info():
    """ General information about VRAM """
    # TODO: check if ´nvidia-smi´ is installed
    # TODO: Enable multi-gpu setup i.e. cuda:0, cuda:1 ...

    # It's not necessary to understand this, it's just an extraction of info from Nvidia's API
    def get_info(command):
        assert command in ["free", "total"]
        command = f"nvidia-smi --query-gpu=memory.{command} --format=csv"
        info = output_to_list(_subprocess.check_output(command.split()))[1:]
        values = [int(x.split()[0]) for i, x in enumerate(info)]
        return values[0]
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    # Format info in a readable way
    free_vram = get_info("free")
    total_vram = get_info("total")
    return {"GPU": _torch.cuda.get_device_properties(0).name,
            "free": free_vram,
            "used": total_vram - free_vram,
            "total": total_vram
            }


def get_gpu_info():
    """ Most useful things about the GPU """
    return {"name": _torch.cuda.get_device_properties(0).name,
            "major": _torch.cuda.get_device_properties(0).major,
            "minor": _torch.cuda.get_device_properties(0).minor,
            "total_memory [MB]": int(_torch.cuda.get_device_properties(0).total_memory / 10 ** 6),
            "multi_processor_count": _torch.cuda.get_device_properties(0).multi_processor_count
            }


def get_screen_dim(WxH=True):
    """ Current screen dimensions in width X height or vice versa """

    # Get screen info through tkinter
    root = _Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    return (width, height) if WxH else (height, width)


def get_os():
    return _platform.platform()


def on_windows():
    return _os.name == "nt"


# TODO add RAM info i.e. amount and type
def get_computer_info():
    """ Most useful things about the computer in general e.g. ram and gpu info """
    uname = _platform.uname()

    print("=" * 40, "System Information", "=" * 40)
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    print(f"GPU: {get_gpu_info()['name']} - {round(get_gpu_info()['total_memory [MB]'])} MB VRAM")
    print("=" * 100)


__all__ = [
    "windows_illegal_file_name_character",
    "get_vram_info",
    "get_gpu_info",
    "get_screen_dim",
    "get_os",
    "on_windows",
    "get_computer_info"
]
