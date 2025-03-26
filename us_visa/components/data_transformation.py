#data_transformation.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler, LabelEncoder

from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.utils.target_value_mapping import TargetValueMapping
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initializes DataTransformation with required configurations and artifacts.
        """
        try:
            logging.info("Initializing DataTransformation class.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.target_value_mapping = TargetValueMapping()
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def _get_data_transformer(self) -> Pipeline:
        try:
            logging.info("Creating data transformation pipeline.")
            
            # Get columns from schema config
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            
            # Verify no overlapping columns between transformation groups
            all_transform_cols = oh_columns + or_columns + transform_columns
            
            if len(set(all_transform_cols)) != len(all_transform_cols):
                duplicates = [x for x in all_transform_cols if all_transform_cols.count(x) > 1]
                raise ValueError(f"Duplicate columns in transformation groups: {duplicates}")

            # Log transformation details
            logging.info(f"OneHotEncoding columns: {oh_columns}")
            logging.info(f"OrdinalEncoding columns: {or_columns}")
            logging.info(f"Transforming & Scaling columns: {transform_columns}")

            # Define transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ('oh', OneHotEncoder(handle_unknown='ignore', drop='first'), oh_columns),
                    ('or', OrdinalEncoder(), or_columns),
                    ('num', Pipeline([
                        ('pt', PowerTransformer()),  # Power transform
                        ('scaler', StandardScaler()) # Then standardize
                    ]), transform_columns)
                ],
                remainder='drop'
            )

            return Pipeline(steps=[('preprocessor', preprocessor)])
            
        except Exception as e:
            logging.error(f"Error creating data transformer: {str(e)}")
            raise CustomException(e, sys)

    def _add_company_age(self, df: pd.DataFrame) -> pd.DataFrame:
        df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
        return df.drop(columns=['yr_of_estab'])  # Drop here instead of in drop_columns


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Performs data transformation on training, validation, and testing datasets.

        Returns:
            DataTransformationArtifact: Artifact containing transformed data paths and objects.
        """
        try:
            logging.info("**********DATA TRANSFORMATION**********")
            # Load datasets
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            val_df = pd.read_csv(self.data_ingestion_artifact.val_file_path)

            # Add company age
            train_df = self._add_company_age(train_df)
            test_df = self._add_company_age(test_df)
            val_df = self._add_company_age(val_df)
            logging.info("Added company age column to datasets.")   

            # Drop columns
            train_df = drop_columns(train_df, self._schema_config['drop_columns'])
            test_df = drop_columns(test_df, self._schema_config['drop_columns'])
            val_df = drop_columns(val_df, self._schema_config['drop_columns'])
            logging.info("Dropped columns from dataset.")

            # Log data shapes
            
            logging.info(f"Training Data Shape after feature selection and extraction: {train_df.shape}")
            logging.info(f"Testing Data Shape after feature selection and extraction: {test_df.shape}")
            logging.info(f"Validation Data Shape after feature selection and extraction: {val_df.shape}")

            # Log columns
            logging.info(f"Train Data Columns after feature selection and extraction: {train_df.columns.tolist()}")
            logging.info(f"Train Data Columns after feature selection and extraction: {test_df.columns.tolist()}")
            logging.info(f"Train Data Columns after feature selection and extraction: {val_df.columns.tolist()}")


            # Split features and target
            input_train, target_train = train_df.drop(TARGET_COLUMN, axis=1), train_df[TARGET_COLUMN]
            input_test, target_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            input_val, target_val = val_df.drop(TARGET_COLUMN, axis=1), val_df[TARGET_COLUMN]

            # Encode target
            target_train_enc = self.target_value_mapping.encode(target_train)
            target_test_enc = self.target_value_mapping.encode(target_test)
            target_val_enc = self.target_value_mapping.encode(target_val)

            # Transform features
            transformation_pipeline = self._get_data_transformer()
            input_train_trans = transformation_pipeline.fit_transform(input_train)
            input_test_trans = transformation_pipeline.transform(input_test)
            input_val_trans = transformation_pipeline.transform(input_val)

            # Combine features and target
            train_arr = np.c_[input_train_trans, target_train_enc]
            test_arr = np.c_[input_test_trans, target_test_enc]
            val_arr = np.c_[input_val_trans, target_val_enc]

            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_val_file_path, val_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, transformation_pipeline)

            logging.info("Data transformation completed successfully.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_val_file_path=self.data_transformation_config.transformed_val_file_path
            )
        


        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_artifact = DataIngestionArtifact(
        trained_file_path="artifacts/data_ingestion/ingested/train.csv",
        test_file_path="artifacts/data_ingestion/ingested/test.csv",
        val_file_path="artifacts/data_ingestion/ingested/validation.csv"
    )

    data_validation_artifact = DataValidationArtifact(
        validation_status=True,  # Use an actual boolean value
        message="Validation successful",
        drift_report_file_path="artifacts/data_validation/drift_report/validation_report.yaml"
    )

    data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=DataTransformationConfig,
                                                     data_validation_artifact=data_validation_artifact)
    data_transformation_artifact = data_transformation.initiate_data_transformation()