import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import CustomException
from us_visa.logger import logging
from us_visa.data_access.import_data import ImportData

class DataIngestion:
    """
    DataIngestion is responsible for:
    1. Exporting data from MongoDB into a CSV file stored in the feature store.
    2. Splitting the dataset into training, validation, and testing subsets.

    Attributes:
        data_ingestion_config (DataIngestionConfig): Configuration settings for data ingestion.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def fetch_data_from_mongodb(self) -> DataFrame:
        """
        Fetches data from MongoDB and returns it as a pandas DataFrame.

        Returns:
            DataFrame: The fetched dataset as a pandas DataFrame.

        Raises:
            CustomException: If an error occurs during data fetching.
        """
        try:
            logging.info("Fetching data from MongoDB.")
            usvisa_data = ImportData()
            dataframe = usvisa_data.import_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Fetched data shape: {dataframe.shape}")
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, dataframe: DataFrame) -> None:
        """
        Exports the given DataFrame to a CSV file in the feature store directory.

        Args:
            dataframe (DataFrame): The dataset to be exported.

        Raises:
            CustomException: If an error occurs during the export process.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving data to feature store at: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("Data saved to feature store successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_val_test(self, dataframe: DataFrame) -> None:
        """
        Splits the dataset into training, validation, and testing sets and saves them to CSV files.

        Args:
            dataframe (DataFrame): The dataset to be split.

        Raises:
            CustomException: If an error occurs during the splitting process.
        """
        logging.info("Starting train-validation-test split of the dataset.")

        try:
            # Split into train and temp (validation + test) first
            train_set, temp_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_temp_split_ratio, random_state=42
            )

            # Split temp into validation and test
            val_set, test_set = train_test_split(
                temp_set, test_size=self.data_ingestion_config.val_test_split_ratio, random_state=42
            )

            logging.info(f"Train set shape: {train_set.shape}, Validation set shape: {val_set.shape}, Test set shape: {test_set.shape}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Saving train, validation, and test datasets.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            val_set.to_csv(self.data_ingestion_config.validation_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Train, validation, and test datasets saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by:
        1. Fetching data from MongoDB.
        2. Exporting data to the feature store.
        3. Splitting the dataset into training, validation, and testing sets.

        Returns:
            DataIngestionArtifact: Contains paths to the training, validation, and testing datasets.

        Raises:
            CustomException: If an error occurs during the ingestion process.
        """
        logging.info("**********DATA INGESTION**********")

        try:
            dataframe = self.fetch_data_from_mongodb()
            logging.info("Data fetched successfully from MongoDB.")

            self.export_data_into_feature_store(dataframe)
            logging.info("Data exported successfully to the feature store.")

            self.split_data_as_train_val_test(dataframe)
            logging.info("Train-validation-test split completed successfully.")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                val_file_path=self.data_ingestion_config.validation_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
