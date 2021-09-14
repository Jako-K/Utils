"""

DESCRIPTION:
Super basic interface with Kaggle's dataset API.

INTENDED USE:
1.) set `project_display_name`, `project_folder_name` and `upload_folder_path`
2.) run this file i.e. _upload_dataset_to_kaggle.py in a Python interpreter

# EXAMPLE:
project_display_name = "Mask Detection Dataset"
project_folder_name = "mask-detection-dataset"
upload_folder_path = "C:/<some_path>/datasets/mask_dataset_yolo"
!python _upload_dataset_to_kaggle.py

"""

import shutil
import os


# Settings
new_dataset = True
project_display_name = "Mask Detection Dataset - 3 classes"
project_folder_name = "mask-detection-dataset-3class"
upload_folder_path = "C:/Users/JK/Desktop/datasets/mask_dataset_yolo"


# Checks
assert project_folder_name.find("_") == -1, "Kaggle doesn't accept underscores, use '-' instead"
assert not os.path.exists("./dataset-metadata.json"), "This should have been deleted"


# Make the json config file kaggle expects
write_string=['{', 
              f'  "title": "{project_display_name}",', 
              f'  "id": "jakobi/{project_folder_name}",', 
               '  "licenses": [{"name": "CC0-1.0"}]',
             '}']

file = open("./dataset-metadata.json", mode="a")
[print(s, file=file) for s in write_string]
file.close()


# Zip the file 
shutil.make_archive(f"./{project_folder_name}", 'zip', upload_folder_path)


# Upload dataset to kaggle (either as a new dataset or as an update to an already existing one)
if new_dataset:
    os.system(f"kaggle datasets create -p ./")
else:
    os.system(f'kaggle datasets version -p ./ -m "new version..."')
    

# Clean up
os.remove("./dataset-metadata.json")
os.remove(f"./{project_folder_name}.zip")


# `dutils.search` expects __all__ to contain everything searchable
__all__ = []