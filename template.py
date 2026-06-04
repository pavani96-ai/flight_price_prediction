import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO, format = '[%(asctime)s:%(message)s:]')

Project_name = "flight_price_prediction"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/{Project_name}/__init__.py",
    f"src/{Project_name}/components/__init__.py",
    f"src/{Project_name}/components/data_ingestion.py",
    f"src/{Project_name}/components/data_validation.py",
    f"src/{Project_name}/components/feature_engineering.py",
    f"src/{Project_name}/components/data_transformation.py",
    f"src/{Project_name}/components/model_training.py",
    f"src/{Project_name}/utils/__init__.py",
    f"src/{Project_name}/utils/common.py",
    f"src/{Project_name}/utils/ml_utils/model.py",
    f"src/{Project_name}/utils/metrics/Regression_metrics.py",
    f"src/{Project_name}/entity/__init__.py",
    f"src/{Project_name}/entity/config_entity.py",
    f"src/{Project_name}/constants/__init__.py",
    f"src/{Project_name}/config/__init__.py",
    f"src/{Project_name}/config/configuration.py",
    f"src/{Project_name}/pipeline/__init__.py",
    f"src/{Project_name}/pipeline/train_pipeline.py",
    f"src/{Project_name}/pipeline/prediction_pipeline.py",
    f"src/{Project_name}/cloud/__init__.py",
    f"src/{Project_name}/logging/__init__.py",
    f"src/{Project_name}/logging/logger.py",
    f"src/{Project_name}/exception/__init__.py",
    f"src/{Project_name}/exception/exception.py",
    "config/config.yaml",
    "params/params.yaml",
    "schema/schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "research/research.ipynb",
    "templates/index.html",
    ".env",
    "test_mongodb.py",
    "push_mongodb.py",
    "__init__.py"


]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)
       
    if filedir!="":
         os.makedirs(filedir,exist_ok=True)
         logging.info(f"Creating directory {filedir} for the file: {filename}")

    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)):
         with open(filepath , "w") as f:
              pass
         logging.info(f"creating empty file : {filepath}")

    else:
         logging.info(f" {filename} already exists")

    
