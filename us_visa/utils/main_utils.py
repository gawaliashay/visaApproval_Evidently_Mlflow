import os
import sys
import numpy as np
import logging
import joblib
import yaml
import pandas as pd
from pandas import DataFrame

from us_visa.exception import CustomException
from us_visa.logger import logging


def read_data(file_path: str) -> pd.DataFrame:
    """
    Read data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data.

    Raises:
        CustomException: If an error occurs while reading the file.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading data from {file_path}: {str(e)}")
        raise CustomException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing the YAML file contents.

    Raises:
        CustomException: If an error occurs while reading the file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            data = yaml.safe_load(yaml_file)
            logging.info(f"Successfully read YAML file: {file_path}")
            return data
    except Exception as e:
        logging.error(f"Error occurred while reading YAML file: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        content (object): Content to write to the file.
        replace (bool): Whether to replace the file if it exists.

    Raises:
        CustomException: If an error occurs while writing the file.
    """
    try:
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


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save object to file: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load an object from a file using joblib.

    Args:
        file_path (str): Path to the file containing the object.

    Returns:
        object: The loaded object.

    Raises:
        CustomException: If an error occurs while loading the object.
    """
    try:
        logging.info(f"Loading object from file: {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Failed to load object from file: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save a NumPy array to a file.

    Args:
        file_path (str): Path to save the array.
        array (np.array): NumPy array to save.

    Raises:
        CustomException: If an error occurs while saving the array.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info(f"Successfully saved array to: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving numpy array to: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load a NumPy array from a file.

    Args:
        file_path (str): Path to the file containing the array.

    Returns:
        np.array: The loaded NumPy array.

    Raises:
        CustomException: If an error occurs while loading the array.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
        logging.info(f"Successfully loaded array from: {file_path}")
        return array
    except Exception as e:
        logging.error(f"Error occurred while loading numpy array from: {file_path}", exc_info=True)
        raise CustomException(e, sys) from e


def save_dataframe_to_csv(file_path: str, dataframe: pd.DataFrame) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        file_path (str): Path to save the CSV file.
        dataframe (pd.DataFrame): DataFrame to save.

    Raises:
        CustomException: If an error occurs while saving the DataFrame.
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


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop specified columns from a DataFrame.

    Args:
        df (DataFrame): The DataFrame to drop columns from.
        cols (list): List of columns to drop.

    Returns:
        DataFrame: The DataFrame with the specified columns dropped.

    Raises:
        CustomException: If an error occurs while dropping columns.
    """
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info(f"Successfully dropped columns: {cols}.")
        return df
    except Exception as e:
        logging.error(f"Error occurred while dropping columns: {cols}", exc_info=True)
        raise CustomException(e, sys) from e