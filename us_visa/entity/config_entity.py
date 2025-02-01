import os
from datetime import datetime
from dataclasses import dataclass
from us_visa.constants import *


# Generate a timestamp to append to directories for uniqueness
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# TrainingPipelineConfig: Main configuration class for the entire pipeline
@dataclass
class TrainingPipelineConfig:
    """
    Configuration for the training pipeline.
    
    Attributes:
        pipeline_name (str): Name of the pipeline.
        artifact_dir (str): Path to the directory where artifacts are saved.
        timestamp (str): Timestamp for uniqueness in file names.
    """
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

# Instantiate the training pipeline configuration
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


# DataIngestionConfig: Configuration related to data ingestion
@dataclass
class DataIngestionConfig:
    """
    Configuration for the data ingestion stage.
    
    Attributes:
        data_ingestion_dir (str): Path to the data ingestion directory.
        feature_store_file_path (str): Path to the feature store file.
        training_file_path (str): Path to the training data.
        testing_file_path (str): Path to the test data.
        validation_file_path (str): Path to the validation data.
        train_temp_split_ratio (float): Ratio for training-validation split.
        val_test_split_ratio (float): Ratio for validation-test split.
        collection_name (str): Name of the data collection.
    """
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    validation_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, VALIDATION_FILE_NAME)
    train_temp_split_ratio: float = DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO
    val_test_split_ratio: float = DATA_INGESTION_VAL_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


# DataValidationConfig: Configuration for data validation
@dataclass
class DataValidationConfig:
    """
    Configuration for the data validation stage.
    
    Attributes:
        data_validation_dir (str): Path to the data validation directory.
        drift_report_file_path (str): Path to the drift report file (if applicable).
    """
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
    )


# DataTransformationConfig: Configuration for data transformation
@dataclass
class DataTransformationConfig:
    """
    Configuration for the data transformation stage.
    
    Attributes:
        data_transformation_dir (str): Path to the data transformation directory.
        transformed_train_file_path (str): Path to the transformed training data.
        transformed_test_file_path (str): Path to the transformed test data.
        transformed_val_file_path (str): Path to the transformed validation data.
        transformed_object_file_path (str): Path to the saved transformation object.
    """
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRANSFORMED_TRAIN_FILE_NAME)
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRANSFORMED_TEST_FILE_NAME)
    transformed_val_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRANSFORMED_VAL_FILE_NAME)
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    """
    Configuration for the model training stage.
    
    Attributes:
        model_trainer_dir (str): Path to the model trainer directory.
        best_model_details_file_path (str): Path to the best model details file.
        trained_model_file_path (str): Path to the trained model file.
        expected_accuracy (float): The minimum expected accuracy for the model.
        model_config_file_path (str): Path to the model configuration file.
    """
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    best_model_details_file_path: str = os.path.join(model_trainer_dir, BEST_MODEL_DETAILS_FILE_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, TRAINED_MODEL_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

# ModelEvaluationConfig: Configuration for model evaluation
@dataclass
class ModelEvaluationConfig:
    """
    Configuration for the model evaluation stage.
    
    Attributes:
        model_evaluation_dir (str): Path to the model evaluation directory.
        evaluation_metrics_file_path (str): Path to the evaluation metrics file.
        existing_model_path (str): Path to the existing model to be compared.
    """
    model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME)
    evaluation_metrics_file_path: str = os.path.join(model_evaluation_dir, METRICS_RESULTS_FILE_NAME)
    existing_model_path: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_DIR_NAME, PRODUCTION_MODEL_FILE_NAME)

