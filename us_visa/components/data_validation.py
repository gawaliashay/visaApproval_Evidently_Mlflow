import json
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset.data_drift import DataDriftPreset
from pandas import DataFrame
from us_visa.exception import CustomException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_data, read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys) from e


    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = [
                column for column in self._schema_config["numerical_columns"] if column not in dataframe_columns
            ]
            missing_categorical_columns = [
                column for column in self._schema_config["categorical_columns"] if column not in dataframe_columns
            ]

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise CustomException(e, sys) from e

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> dict:
        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=reference_df, current_data=current_df)
            report = data_drift_report.as_dict()

            # Log detailed drift report for debugging
            logging.info("Detailed drift report:")
            #logging.info(json.dumps(report, indent=4))

            # Extract essential metrics
            drift_info = {
                "dataset_drift": report["metrics"][0]["result"].get("dataset_drift", False),
                "drift_share": report["metrics"][0]["result"].get("drift_share", None),
                "number_of_features": report["metrics"][0]["result"].get("number_of_columns", 0),  # Fix key
                "number_of_drifted_features": report["metrics"][0]["result"].get("number_of_drifted_columns", 0),  # Fix key
                "share_of_drifted_features": report["metrics"][0]["result"].get("share_of_drifted_columns", None),  # Fix key
                "drifted_features": [
                        column for column, details in report["metrics"][1]["result"].get("drift_by_columns", {}).items()
                        if details.get("drift_detected", False)
                    ],  # Extract drifted feature names                
                "feature_importance": None,  # Placeholder (adjust if needed)
                "p_values": [],  # Placeholder (adjust if needed)
                "statistical_tests": {},  # Placeholder (adjust if needed)
                "target_drift": None,  # Placeholder (adjust if needed)
                "distribution_difference": {},  # Placeholder (adjust if needed)
                "drift_threshold": report["metrics"][1]["result"].get("stattest_threshold", None),  # New
            }

            # Save a simplified report
            logging.info("Simplified drift info:")
            logging.info(json.dumps(drift_info, indent=4))
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=drift_info)

            logging.info(f"Drift report saved to {self.data_validation_config.drift_report_file_path}")

            return drift_info
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("**********DATA VALIDATION**********")

            train_df = read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = read_data(file_path=self.data_ingestion_artifact.test_file_path)

            # Validate number of columns
            if not self.validate_number_of_columns(dataframe=train_df):
                validation_error_msg += "Columns are missing in training dataframe. "
            if not self.validate_number_of_columns(dataframe=test_df):
                validation_error_msg += "Columns are missing in test dataframe. "

            # Validate existence of required columns
            if not self.is_column_exist(df=train_df):
                validation_error_msg += "Required columns are missing in training dataframe. "
            if not self.is_column_exist(df=test_df):
                validation_error_msg += "Required columns are missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_info = self.detect_dataset_drift(train_df, test_df)
                if drift_info["dataset_drift"]:
                    logging.info("Drift detected.")
                    validation_error_msg = "Drift detected."
                else:
                    logging.info("No drift detected.")
            else:
                logging.info(f"Validation errors: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
