import sys
import os
import json
import numpy as np
from typing import Tuple

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from us_visa.entity.optuna_tuner import tune_model

from us_visa.exception import CustomException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from us_visa.entity.estimator import USvisaModel

# Import the MlflowExperimentEvaluation class
from us_visa.entity.mlflow_experiment_evaluation import MlflowExperimentEvaluation


class ModelTrainer:
    """
    A class to handle model training, hyperparameter tuning, and evaluation.
    """

    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer class.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Contains file paths of transformed data.
            model_trainer_config (ModelTrainerConfig): Contains training-related configurations.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def tune_and_evaluate_model_with_optuna(self, train: np.ndarray, val: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Performs hyperparameter tuning using Optuna and evaluates the best model.

        Args:
            train (np.ndarray): Training dataset as a NumPy array.
            val (np.ndarray): Validation dataset as a NumPy array.

        Returns:
            Tuple[object, ClassificationMetricArtifact]: Best model object and classification metrics artifact.

        Raises:
            CustomException: If any error occurs during model tuning or evaluation.
        """
        try:
            logging.info("Initiating HYPERPARAMETER TUNING using OPTUNA.")

            # Split train and validation datasets into features and targets
            x_train, y_train = train[:, :-1], train[:, -1]
            x_val, y_val = val[:, :-1], val[:, -1]
            logging.debug(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")

            # Perform hyperparameter tuning
            best_model, best_params, best_score = tune_model(
                X_train=x_train,
                y_train=y_train,
                model_config_path=self.model_trainer_config.model_config_file_path
            )
            logging.info(f"Optuna tuning completed. Best Model: {best_model}, Params: {best_params}, Score: {best_score}")

            # Evaluate the best model on the validation set
            y_val_pred = best_model.predict(x_val)
            accuracy = accuracy_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred)
            recall = recall_score(y_val, y_val_pred)
            logging.info(f"Model evaluation metrics: Accuracy={accuracy}, F1={f1}, Precision={precision}, Recall={recall}")

            # Create a metrics artifact
            metric_artifact = ClassificationMetricArtifact(
                accuracy=accuracy,
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            return best_model, metric_artifact

        except Exception as e:
            logging.error(f"Error in get_model_object_and_report: {str(e)}")
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process by loading data, performing model tuning, and saving the best model.

        Returns:
            ModelTrainerArtifact: Contains the trained model path and metrics artifact.

        Raises:
            CustomException: If any error occurs during the training process.
        """
        logging.info("********** MODEL TRAINER **********")
        try:
            # Load transformed training and validation data
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            val_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_val_file_path)
            logging.debug(f"Training data shape: {train_arr.shape}, Validation data shape: {val_arr.shape}")

            # Get the best model and its metrics
            best_model, metric_artifact = self.tune_and_evaluate_model_with_optuna(train=train_arr, val=val_arr)

            # Save best model details to a JSON file
            best_model_details = {
                "BestModel": str(best_model),                   # Convert the model object to a string
                "metric_artifact": metric_artifact.__dict__     # Convert the metric artifact object to a dictionary
            }

            # Create the directory to save the best model details
            dir_path = os.path.dirname(self.model_trainer_config.best_model_details_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the best model details to a file
            with open(self.model_trainer_config.best_model_details_file_path, "w") as f:
                json.dump(best_model_details, f, indent=4)

            # Ensure the model meets the expected accuracy threshold
            if metric_artifact.accuracy < self.model_trainer_config.expected_accuracy:
                logging.error("No suitable model found with accuracy above the expected threshold.")
                raise CustomException("No suitable model found with accuracy above the expected threshold.")

            # Load preprocessing object used during transformation
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded successfully.")

            # Create and save the USvisa model object
            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )

            # Save the USvisa model locally using the path from config_entity.py
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)
            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}.")

            # Initialize MLflow experiment evaluation
            mlflow_experiment = MlflowExperimentEvaluation(
                experiment_name="US_Visa_Classification",
                run_name="Model_Training_Run",
                model_name="us_visa_model"
            )

            # Log the experiment and save the best model in MLflow
            mlflow_experiment.run_mlflow_experiment(
                metric_name="accuracy",
                metric_value=metric_artifact.accuracy,
                model=best_model,
                best_model_parameters=best_model.get_params(),
                model_name="us_visa_model",
                dst_path=self.model_trainer_config.trained_model_file_path  # Pass the local save path here
            )

            # Create and return the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                best_model_details_file_path=self.model_trainer_config.best_model_details_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"ModelTrainerArtifact created: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(e, sys) from e