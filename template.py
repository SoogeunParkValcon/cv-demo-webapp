# script for creating directories and file tepmlates for a new project

import os
from pathlib import Path
import logging

# defining logging string
# for log level = INFO
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# defining the project name (this will also be the name of the root directory)
project_name = 'yolo_webapp'

list_of_files = [
    '.github/workflows/.gitkeep',
    'data/.gitkeep',
    f'{project_name}/__init__.py',
    f'{project_name}/components/__init__.py',
    f'{project_name}/constant/__init__.py',
    f'{project_name}/constant/training_pipeline/__init__.py',
    f'{project_name}/constant/training_pipeline/application.py',
    f'{project_name}/entity/__init__.py',
    f'{project_name}/entity/config_entity.py',
    f'{project_name}/entity/artifacts_entity.py',
    f'{project_name}/exception/__init__.py',
    f'{project_name}/logger/__init__.py',
    f'{project_name}/pipeline/__init__.py',
    f'{project_name}/utils/__init__.py',
    'templates/index.html',
    'app.py',
    'setup.py',
    'tests.py',
    'Dockerfile',
    'requirements.txt',
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if(not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")

    else:
        logging.info(f"{filename} is already created")