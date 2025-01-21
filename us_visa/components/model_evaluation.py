from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from us_visa.exception import CustomException
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.logger import logging
import pandas as pd
import sys
from typing import Optional
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.label_encoder = LabelEncoder()
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function compares the trained model with the production model using F1 score.

        Output      :   Returns an EvaluateModelResponse with evaluation results.
        On Failure  :   Logs and raises an exception.
        """
        try:
            # Load test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']

            # Separate features and target
            x_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # Encode target labels
            y_test = self.label_encoder.fit_transform(y_test)

            # Retrieve F1 score of the trained model
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            trained_model_name = self.model_trainer_artifact.trained_model_file_path

            # Initialize comparison variables
            best_model_f1_score = None

            # Determine if the trained model is better
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = trained_model_f1_score > tmp_best_model_score
            difference = trained_model_f1_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

            # Log detailed evaluation results
            logging.info(f"Evaluation Details:\n"
                         f"Trained Model Name: {trained_model_name}\n"
                         f"Trained Model F1 Score: {trained_model_f1_score}\n"
                         f"Best Model F1 Score: {best_model_f1_score if best_model_f1_score is not None else 'N/A'}\n"
                         f"Is Model Accepted: {is_model_accepted}\n"
                         f"Score Difference: {difference}")

            return result

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function executes the model evaluation process.

        Output      :   Returns a ModelEvaluationArtifact with evaluation details.
        On Failure  :   Logs and raises an exception.
        """
        try:
            # Perform evaluation
            evaluate_model_response = self.evaluate_model()

            # Create artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
