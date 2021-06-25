import shutil
import os


# Settings
new_dataset = True
project_display_name = "Sets Data Augmetation - Validation"
project_folder_name = "sets-data-augmentation-validation"
upload_folder_path = r"C:\Users\JK\Desktop\results"


# Checks
assert project_folder_name.find("_") == -1, "Kaggle don't accept underscores"
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