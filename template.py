import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - %(message)s")


PROJECT_NAME = "car_price_prediction"

list_of_files = [
    "app.py",
    "setup.py",
    "requirements.txt",
    f"src/__init__.py",
    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/utlis/__init__.py",
    f"src/{PROJECT_NAME}/constant/__init__.py",
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/components/data_ingestion.py",
    f"src/{PROJECT_NAME}/components/data_transformation.py",
    f"src/{PROJECT_NAME}/components/model_trainer.py",
    f"src/{PROJECT_NAME}/pipelines/__init__.py",
    f"src/{PROJECT_NAME}/pipelines/training_pipeline.py",
    f"src/{PROJECT_NAME}/pipelines/predicrion_pipeline.py",
    f"src/logger/__init__.py",
    f"src/exception/__init__.py",
    'notebooks/eda.ipynb'
]


for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"The Directory {file_dir} is created for file:{file_name}")

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            logging.info(f"The file:{file_name} is created in {file_dir}")
            pass

    else:
        logging.info(f"The file:{file_name} already exits")
