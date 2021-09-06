from importlib.machinery import SourceFileLoader

foo = SourceFileLoader("colors", "./colors.py").load_module()
