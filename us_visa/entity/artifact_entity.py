from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    """
    Artifact for data ingestion stage.
    
    Attributes:
        trained_file_path (str): Path to the file containing the training data.
        val_file_path (str): Path to the file containing the validation data.
        test_file_path (str): Path to the file containing the test data.
    """
    trained_file_path: str
    val_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Artifact for data validation stage.
    
    Attributes:
        validation_status (bool): Indicates if data validation passed or failed.
        message (str): A detailed message or reason for validation status.
        drift_report_file_path (str): Path to the report on data drift (if applicable).
    """
    validation_status: bool
    message: str
    drift_report_file_path: Optional[str] = None


@dataclass
class DataTransformationArtifact:
    """
    Artifact for data transformation stage.
    
    Attributes:
        transformed_object_file_path (str): Path to the saved transformation object.
        transformed_train_file_path (str): Path to the transformed training data.
        transformed_test_file_path (str): Path to the transformed test data.
        transformed_val_file_path (str): Path to the transformed validation data.
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_val_file_path: str


@dataclass
class ClassificationMetricArtifact:
    """
    Artifact for classification metrics.
    
    Attributes:
        accuracy (float): Accuracy score of the model.
        f1_score (float): F1 score of the model.
        precision_score (float): Precision score of the model.
        recall_score (float): Recall score of the model.
    """
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Artifact for model training stage.
    
    Attributes:
        trained_model_file_path (str): Path to the trained model file.
        best_model_details_file_path (str): Path to the details of the best model (if applicable).
        metric_artifact (ClassificationMetricArtifact): Metrics artifact containing classification metrics.
    """
    trained_model_file_path: str
    best_model_details_file_path: str
    metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    """
    Artifact for model evaluation stage.
    
    Attributes:
        is_model_accepted (bool): Indicates if the model is accepted based on evaluation metrics.
        test_f1_score (float): F1 score for the test dataset.
        test_accuracy (float): Accuracy score for the test dataset.
        evaluation_metrics_file_path (str): Path to the saved evaluation metrics report.
        existing_model_path (str): Path to the existing model that is being compared.
        evaluated_model_path (str): Path to the new evaluated model.
        current_model_f1_score (float): F1 score of the current model after evaluation.
        existing_model_f1_score (float): F1 score of the existing model (if available).
    """
    is_model_accepted: bool
    test_f1_score: float
    test_accuracy: float
    evaluation_metrics_file_path: str
    existing_model_path: str
    evaluated_model_path: str
    current_model_f1_score: float
    existing_model_f1_score: float
