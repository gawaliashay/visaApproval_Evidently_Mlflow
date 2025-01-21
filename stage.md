# US Visa Data Processing and Machine Learning Pipeline

## Overview
This project provides an end-to-end solution for processing US Visa data and building a machine learning pipeline. The pipeline comprises multiple stages including data ingestion, transformation, validation, model training, evaluation, and deployment. It incorporates modular components, reusable configurations, and custom logging and error handling for seamless execution.

---

## Project Setup and Configuration

### 1. **Setup Virtual Environment and Dependencies**
- Create a Conda virtual environment.
- Install all required dependencies listed in `requirements.txt`.

### 2. **Project Structure**
- Use `template.py` to define the project’s folder structure for logical organization of pipeline components.

### 3. **Configuration and Packaging**
- Manage project dependencies in `requirements.txt`.
- Use `pyproject.toml` for packaging and sharing the project.

### 4. **Logging and Error Handling**
- Implement logging configurations in `logger.py`.
- Set up custom exception handling mechanisms in `exception.py`.

### 5. **Constants and Entities**
- Use `constants.py` for defining reusable constants like file paths and default values.
- Create `config_entity.py` for specifying input configurations to each pipeline stage.
- Define outputs from pipeline stages in `artifact_entity.py`.

### 6. **Database and Data Handling**
- Establish a MongoDB connection using `mongo_db_connection.py`.
- Use `data_dump.py` to upload raw CSV data to MongoDB.
- Fetch and export data back to CSV using `data_access.usvisa_data.py`.

---

## Pipeline Components

### 1. **Data Ingestion**
- **Purpose**: Fetch data from MongoDB, split it into training and testing sets, and save as CSV files.
- **Inputs**:
  - MongoDB data fetched using configurations from `config_entity.py`.
  - Constants like file paths from `constants.py`.
- **Outputs**:
  - Training and testing datasets saved locally.
  - Artifact returned as `DataIngestionArtifact`.
- **Functions**:
  - `export_data_into_feature_store`: Fetches data from MongoDB and saves it as a CSV file in feature store.
  - `split_data_as_train_test`: Splits data into training/testing sets and saves them.
  - `initiate_data_ingestion`: Coordinates ingestion and returns artifacts.

### 2. **Data Validation**
- **Purpose**: Ensure data integrity through schema validation and detect dataset drift.
- **Inputs**:
  - Data schema and configurations from `config_entity.py`.
  - Training/testing datasets.
- **Outputs**:
  - Validation results and drift reports saved as YAML files.
  - Artifact returned as `DataValidationArtifact`.
- **Functions**:
  - `validate_number_of_columns`: Checks dataset for correct column count.
  - `is_column_exist`: Verifies presence of required columns.
  - `detect_dataset_drift`: Generates drift reports.
  - `initiate_data_validation`: Executes all validation steps and returns artifacts.

### 3. **Data Transformation**
- **Purpose**: Transform data with preprocessing, apply balancing techniques, and save transformed datasets.
- **Inputs**:
  - Train/test datasets from Data Ingestion.
  - Preprocessing configurations from `config_entity.py`.
- **Outputs**:
  - Transformed datasets and preprocessor object.
  - Artifact returned as `DataTransformationArtifact`.
- **Functions**:
  - `get_data_transformer_object`: Creates a preprocessing pipeline (e.g., scaling, encoding).
  - `initiate_data_transformation`: Applies transformations and saves artifacts.

### 4. **Model Trainer**
- **Purpose**: Train machine learning models and evaluate their performance.
- **Inputs**:
  - Transformed datasets.
  - Model configurations from `config_entity.py`.
- **Outputs**:
  - Trained model and evaluation metrics.
  - Artifact returned as `ModelTrainerArtifact`.
- **Functions**:
  - `get_model_object_and_report`: Trains models and generates metrics.
  - `initiate_model_trainer`: Saves the best model and preprocessing object.

### 5. **Model Evaluation**
- **Purpose**: Compare the trained model against the production model and decide acceptance.
- **Inputs**:
  - Trained model.
  - Production model fetched from S3.
  - Evaluation criteria from `config_entity.py`.
- **Outputs**:
  - Evaluation results and decision on model acceptance.
  - Artifact returned as `ModelEvaluationArtifact`.
- **Functions**:
  - `get_best_model`: Retrieves the production model from S3.
  - `evaluate_model`: Compares F1 scores of new and production models.
  - `initiate_model_evaluation`: Returns evaluation artifacts.

### 6. **Model Pusher**
- **Purpose**: Deploy the accepted model to S3 for production use.
- **Inputs**:
  - Accepted model and deployment configurations.
- **Outputs**:
  - Confirmation of successful model deployment.
  - Artifact returned as `ModelPusherArtifact`.
- **Functions**:
  - `initiate_model_pusher`: Uploads the accepted model to S3.

---

## key Files and Configurations needs to be updated/checked while creating every component :

### 1. **Configuration and Artifacts**
- `config_entity.py`: Defines configurations for each pipeline stage.
- `artifact_entity.py`: Specifies output artifacts for all stages.

### 2. **Constants**
- `constants.py`: Stores paths and reusable constants.

### 3. **Environment Variables**
- `.env`: Stores environment-specific variables like credentials and paths.

### 4. **Dependencies**
- `requirements.txt`: Lists all required Python libraries.

### 5. **Packaging**
- `pyproject.toml`: Specifies project metadata for packaging and distribution.

### 6. **Logging and Exception Handling**
- `logger.py`: Configures custom logging.
- `exception.py`: Defines reusable error-handling mechanisms.

---

## Summary
This modular pipeline efficiently processes US Visa data through robust components for ingestion, transformation, validation, training, evaluation, and deployment. It is designed to facilitate maintainable and scalable machine learning workflows for IT professionals.
