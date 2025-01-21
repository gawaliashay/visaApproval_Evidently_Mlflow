import os
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.exception import CustomException
from us_visa.logger import logging
from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.utils.main_utils import save_object, read_yaml_file, save_numpy_array_data, save_dataframe_to_csv
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

class DataTransformation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initialize DataTransformation with the required configuration and artifacts.
        """
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformation pipeline object.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # Load columns from schema configuration
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            # Define transformers
            oh_transformer = OneHotEncoder(handle_unknown='ignore')
            or_transformer = OrdinalEncoder()
            pt_transformer = PowerTransformer()
            scaler = StandardScaler()

            # Create ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('oh', oh_transformer, oh_columns),
                    ('or', or_transformer, or_columns),
                    ('pt', pt_transformer, transform_columns),
                    ('scaler', scaler, num_features)
                ],
                remainder='drop'
            )

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)
            ])

            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Loading training and test data")
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"train_df shape: {train_df.shape}, test_df shape: {test_df.shape}")
            
            # Generate company_age column if required
            train_df['company_age'] = CURRENT_YEAR - train_df['yr_of_estab']
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info(f"input_feature_train_df shape: {input_feature_train_df.shape}, input_feature_test_df shape: {input_feature_test_df.shape}")   
            # Encode target labels
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info(f"target_feature_train_arr shape: {target_feature_train_arr.shape}, target_feature_test_arr shape: {target_feature_test_arr.shape}")   
            # Transform input features
            logging.info("Getting data transformer object")
            transformation_pipeline = self.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)
            logging.info(f"input_feature_train_arr shape: {input_feature_train_arr.shape}, input_feature_test_arr shape: {input_feature_test_arr.shape}")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_arr)]

            logging.info("Saving transformed data")
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=transformation_pipeline)

            logging.info("Data transformation completed successfully")

            # Create and return DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info(f"DataTransformationArtifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
