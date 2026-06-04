import os
import yaml
import sys
import numpy as np
import pandas as pd
import pickle
from src.flight_price_prediction.logging.logger import logging
from pathlib import Path
from src.flight_price_prediction.exception.exception import CustomException

import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from typing import Any, Union
from box.exceptions import BoxValueError


def create_directory(path: Union[Path, str], is_file: bool = True) -> object:
    """
    Ensure the directory exists for a file or directory path.
    Args:
        path (Path | str): file or directory path
        is_file (bool): when True, create the parent directory for a file path;
            when False, create the directory path itself.
    """
    try:
        path_obj = Path(path)
        directory = path_obj.parent if is_file else path_obj
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """
    read yaml file and return
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
       ConfigBox: ConfigBox type
       
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded succesfully ")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations    
def write_yaml_file(file_path: str, content: object, replace: bool =False) -> object:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        create_directory(file_path)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    create list of directories
    Args:
        path_to_directories (list): list of path of directories
        verbose(bool,optional): ignore if multiple dirs is to be created. Defaults to False
     """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"created directory at: {path}")
    except Exception as e:
        raise CustomException(e, sys)
    
@ensure_annotations
def save_json(path: Path, data: dict):
    """
    save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
        
    """
    try:
        create_directory(path)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"json file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)
    
@ensure_annotations
def load_json(path:Path) -> ConfigBox:
    """
    load json files data
    Args:
        path(Path): path to json file
    Returns:
        ConfigBox: data as class attributes instead of dict"""
    
    try:
        with open(path) as f:
            content = json.load(f)
        logging.info(f"json file loaded succesfully from : {path}")
        return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)
    
@ensure_annotations
def save_bin(data: Any, path:Path):
    """
    save binary file
    Args:
        data(Any): data to be saved as binary
        path(Path): path to binary file
    """
    try:
        logging.info("Entered the save_bin method of common utils class")
        create_directory(path)
        pickle.dump(value=data, filename=path)
        logging.info(f"binary file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)
@ensure_annotations
def load_bin(path:Path) -> Any:
    """
    load binary data

    Args: 
        path(Path): path to binary file

    Returns:
        Any: object stored in the file

    """
    try:
        if not os.path.exists(path):
            raise Exception(f"The file: {path} is not exists")
        with open(path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def save_data(data: pd.DataFrame, path: Path):
    create_directory(path)
    data.to_csv(path, index=False, header=True)
    logging.info(f"Data saved to: {path}")

def save_numpy_array(file_path: str, array: np.array):
    try:
        create_directory(file_path)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) 

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) 