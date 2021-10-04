from setuptools import setup, find_packages


setup(
    name="dutils",
    version='0.0.1',
    author="Jako-K",
    description='Helper functions for data science',
    packages=find_packages(),
    install_requires=[
        "jupyter_core", 
        "pandas", 
        "torchaudio", 
        "requests", 
        "pydicom", 
        "opencv_python_headless", 
        "matplotlib", 
        "torch", 
        "numpy", 
        "ipython", 
        "jupyter", 
        "Pillow",
        "validators",
        "pynput",
        "wandb"
    ]
)