import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR, SCHEMA_FILE_PATH



from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.utils.main_utils import load_object, save_object, drop_columns, read_yaml_file
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.utils.target_value_mapping import TargetValueMapping
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact,
)


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self.target_value_mapping = TargetValueMapping()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_transformation_and_model(self):
        try:
            logging.info("Loading transformation object and trained model.")
            
            if not os.path.exists(self.data_transformation_artifact.transformed_object_file_path):
                raise FileNotFoundError(
                    f"Transformation pipeline file not found: {self.data_transformation_artifact.transformed_object_file_path}"
                )
            
            if not os.path.exists(self.model_trainer_artifact.trained_model_file_path):
                raise FileNotFoundError(
                    f"Trained model file not found: {self.model_trainer_artifact.trained_model_file_path}"
                )
            
            transformation_pipeline = load_object(self.data_transformation_artifact.transformed_object_file_path)
            current_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            
            return transformation_pipeline, current_model
        except Exception as e:
            raise CustomException(e, sys)
        
    def _add_company_age(self, df: pd.DataFrame) -> pd.DataFrame:
        df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
        return df.drop(columns=['yr_of_estab'])  # Drop here instead of in drop_columns

    def _evaluate_model(self, model, features: np.ndarray, target: np.ndarray):
        try:
            logging.info("Evaluating model.")
            predictions = model.predict(features)
            f1 = f1_score(target, predictions)
            accuracy = accuracy_score(target, predictions)
            return f1, accuracy
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("********** MODEL EVALUATION **********")

            test_file_path = self.data_ingestion_artifact.test_file_path
            if not os.path.exists(test_file_path):
                raise FileNotFoundError(f"Test file does not exist: {test_file_path}")

            logging.info(f"Loading test data from CSV: {test_file_path}")
            test_df = pd.read_csv(test_file_path)

            logging.info(f"Original DataFrame shape: {test_df.shape}")
            logging.info(f"Original DataFrame columns: {list(test_df.columns)}")

            test_df = self._add_company_age(test_df)

            logging.info(f"DataFrame shape after preprocessing: {test_df.shape}")
            logging.info(f"Remaining columns: {list(test_df.columns)}")

            x_test, y_test = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]
            y_test = self.target_value_mapping.encode(y_test)

            logging.info(f"Data types in x_test: {x_test.dtypes}")
            logging.info(f"Sample x_test: {x_test.head().to_string()}")
            logging.info(f"Data types in y_test: {y_test.dtype}")
            logging.info(f"Sample y_test: {y_test[:5]}")

            if not isinstance(x_test, np.ndarray):
                # Ensure the transformation pipeline is applied before converting to NumPy
                transformation_pipeline, current_model = self._load_transformation_and_model()

                logging.info("Applying transformation pipeline on x_test.")
                x_test = transformation_pipeline.transform(x_test)

                logging.info(f"Transformed x_test shape: {x_test.shape}")

                # Convert to NumPy only after transformation
                x_test = np.array(x_test)

            if not isinstance(y_test, np.ndarray):
                y_test = y_test.to_numpy()

            logging.info(f"Converted x_test to NumPy array. Shape: {x_test.shape}, Type: {x_test.dtype}")
            logging.info(f"Converted y_test to NumPy array. Shape: {y_test.shape}, Type: {y_test.dtype}")

            transformation_pipeline, current_model = self._load_transformation_and_model()
            current_model_f1, current_model_accuracy = self._evaluate_model(current_model, x_test, y_test)
            logging.info(f"Current model - F1 Score: {current_model_f1}, Accuracy: {current_model_accuracy}")

            existing_model_path = self.model_evaluation_config.existing_model_path
            existing_model_f1, existing_model_accuracy = None, None
            is_model_accepted = False

            if os.path.exists(existing_model_path):
                logging.info(f"Existing model found at: {existing_model_path}")
                existing_model = load_object(existing_model_path)
                existing_model_f1, existing_model_accuracy = self._evaluate_model(existing_model, x_test, y_test)
                logging.info(f"Existing model - F1 Score: {existing_model_f1}, Accuracy: {existing_model_accuracy}")
                
                if current_model_f1 > existing_model_f1:
                    logging.info("Current model outperforms the existing model.")
                    save_object(existing_model_path, current_model)
                    is_model_accepted = True
                else:
                    logging.info("Existing model performs better or equal.")
            else:
                logging.info("No existing model found. Accepting the current model.")
                save_object(existing_model_path, current_model)
                is_model_accepted = True

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                test_f1_score=current_model_f1,
                test_accuracy=current_model_accuracy,
                evaluation_metrics_file_path=self.model_trainer_artifact.metric_artifact,
                existing_model_path=existing_model_path,
                evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                current_model_f1_score=current_model_f1,
                existing_model_f1_score=existing_model_f1,
            )

            logging.info(f"Model evaluation completed successfully.")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
