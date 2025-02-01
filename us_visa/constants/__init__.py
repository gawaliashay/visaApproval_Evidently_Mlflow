import os
from dotenv import load_dotenv
from datetime import date

# Load environment variables from .env file
load_dotenv()

DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGODB_URL_KEY = os.getenv("MONGODB_URL_KEY")



PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
VALIDATION_FILE_NAME: str = "validation.csv"

TRANSFORMED_TRAIN_FILE_NAME: str = "transformed_train.npy"
TRANSFORMED_TEST_FILE_NAME: str = "transformed_test.npy"
TRANSFORMED_VAL_FILE_NAME: str = "transformed_val.npy"

FILE_NAME: str = "usvisa.csv"
TRAINED_MODEL_FILE_NAME = "trained_model.pkl"
PRODUCTION_MODEL_FILE_NAME = "production_model.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

'''
Data Ingestion retaled constants start with DATA_INGESTION VAR NAME
'''
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO: float = 0.2
DATA_INGESTION_VAL_TEST_SPLIT_RATIO: float = 0.5

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
BEST_MODEL_DETAILS_FILE_NAME: str = "best_model_details.json"


"""
MODEL EVALUATION related constant 
"""
METRICS_RESULTS_FILE_NAME = "metrics_results.json"
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02


APP_HOST = "0.0.0.0"
APP_PORT = 8080