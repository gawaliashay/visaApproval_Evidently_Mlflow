import os
import sys

import numpy as np
import dill
import yaml
import pandas as pd
from pandas import DataFrame

from us_visa.exception import CustomException
from us_visa.logger import logging


def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    try:
        logging.info(f"Attempting to read YAML file from: {file_path}")
        with open(file_path, "rb") as yaml_file:
            data = yaml.safe_load(yaml_file)
            logging.info(f"Successfully read YAML file: {file_path}")
            return data
    except Exception as e:
        logging.error(f"Error occurred while reading YAML file: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        logging.info(f"Attempting to write YAML file to: {file_path}")
        if replace and os.path.exists(file_path):
            logging.info(f"Replacing existing file at: {file_path}")
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logging.info(f"Successfully wrote YAML file to: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing YAML file: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e

def load_object(file_path: str) -> object:
    logging.info(f"Entered the load_object method. Attempting to load object from: {file_path}")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Successfully loaded object from: {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error occurred while loading object from: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    logging.info(f"Attempting to save numpy array to: {file_path}")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info(f"Successfully saved numpy array to: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving numpy array to: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    logging.info(f"Attempting to load numpy array from: {file_path}")
    try:
        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
        logging.info(f"Successfully loaded numpy array from: {file_path}")
        return array
    except Exception as e:
        logging.error(f"Error occurred while loading numpy array from: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e
    

def save_dataframe_to_csv(file_path: str, dataframe: pd.DataFrame):
    """
    Saves a DataFrame to a CSV file after ensuring the directory exists.

    :param file_path: Path to save the CSV file.
    :param dataframe: Pandas DataFrame to save.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the DataFrame
        dataframe.to_csv(file_path, index=False)
        logging.info(f"DataFrame saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {file_path}: {e}")
        raise CustomException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    logging.info(f"Entered the save_object method. Attempting to save object to: {file_path}")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Successfully saved object to: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving object to: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e

def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop the columns from a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info(f"Entered drop_columns method. Attempting to drop columns: {cols}")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info(f"Successfully dropped columns: {cols}")
        return df
    except Exception as e:
        logging.error(f"Error occurred while dropping columns: {cols}", exc_info=True)
        raise CustomException(e, sys) from e