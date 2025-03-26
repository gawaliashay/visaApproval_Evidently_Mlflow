# mlflow_experiment_evaluation.py

import sys
import logging
import os
import platform
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from us_visa.exception import CustomException
from us_visa.utils.main_utils import save_object


class MlflowExperimentEvaluation:
    """
    A class to manage MLflow experiments, evaluate the best model, and generate reports.
    """

    def __init__(self, experiment_name: str, run_name: str, model_name: Optional[str] = None):
        """
        Initialize the ExperimentEvaluation class.

        Args:
            experiment_name (str): Name of the MLflow experiment.
            run_name (str): Name of the MLflow run.
            model_name (str, optional): Name of the model. Defaults to None.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model_name = model_name

        self.best_model_run_id = None
        self.best_model_uri = None
        self.model_path = None
        self.artifact_uri = None

    def get_best_model_run_id(self, experiment_name: str, metric_name: str, maximize: bool = True) -> None:
        """
        Retrieve the run ID of the best model from the MLflow experiment based on a specified metric.

        Args:
            experiment_name (str): Name of the experiment.
            metric_name (str): Name of the metric to evaluate the best model.
            maximize (bool, optional): Whether to maximize the metric. Defaults to True.

        Raises:
            CustomException: If the experiment or runs are not found.
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise CustomException(f"Experiment '{experiment_name}' not found.")

            order = f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=[order])
            
            if runs.empty:
                raise CustomException(f"No runs found for experiment '{experiment_name}'.")

            best_run = runs.iloc[0]
            self.best_model_run_id = best_run.run_id
            self.best_model_uri = f"runs:/{best_run.run_id}/model"
            logging.info(f"Best model run ID retrieved: {self.best_model_run_id}")

        except Exception as e:
            logging.error(f"Error in get_best_model_run_id: {str(e)}")
            raise CustomException(e, sys) from e

    def save_model(self, dst_path: str, artifact_name: str = "model", model_format: str = "pyfunc") -> None:
        """
        Save the best model to the specified destination path.

        Args:
            dst_path (str): Destination path to save the model.
            artifact_name (str, optional): Name of the model artifact. Defaults to "model".
            model_format (str, optional): Format of the model ("pyfunc" or "sklearn"). Defaults to "pyfunc".
        """
        try:
            run = mlflow.get_run(self.best_model_run_id)
            self.artifact_uri = run.info.artifact_uri
            model_uri = f"{self.artifact_uri}/{artifact_name}"

            if model_format == "pyfunc":
                model = mlflow.pyfunc.load_model(model_uri)
            elif model_format == "sklearn":
                model = mlflow.sklearn.load_model(model_uri)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            save_object(file_path=dst_path, obj=model)
            logging.info(f"Model saved successfully to: {dst_path}")

        except Exception as e:
            logging.error(f"Error in save_model: {str(e)}")
            raise CustomException(e, sys) from e

    def create_run_report(self) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Generate a report for the best model run.
        """
        try:
            best_run = mlflow.get_run(self.best_model_run_id)
            best_model_metrics = best_run.data.metrics
            best_model_parameters = best_run.data.params

            report = {
                "best_run_id": best_run.info.run_id,
                "best_run_experiment_id": best_run.info.experiment_id,
                "best_run_artifact_uri": best_run.info.artifact_uri,
                "best_model_name": self.model_name,
                "best_model_parameters": best_model_parameters,
                "best_model_metrics": best_model_metrics,
                "environment": {
                    "python_version": platform.python_version(),
                    "system": platform.system(),
                    "machine": platform.machine()
                }
            }

            logging.info(f"Run report generated for model: {self.model_name}")
            return self.model_name, best_model_parameters, best_model_metrics, report

        except Exception as e:
            logging.error(f"Error in create_run_report: {str(e)}")
            raise CustomException(e, sys) from e

    def run_mlflow_experiment(
        self, 
        metric_name: str, 
        metric_value: float, 
        model: Any, 
        best_model_parameters: Dict[str, Any], 
        model_name: str, 
        dst_path: str,
        X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Execute an MLflow experiment with enhanced logging capabilities.
        """
        try:
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_name=self.run_name):
                # Set basic tags
                mlflow.set_tag("mlflow.runName", self.run_name)
                mlflow.set_tag("model_type", self.model_name)
                
                # Add custom tags if provided
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)

                # Log parameters
                for key, value in best_model_parameters.items():
                    mlflow.log_param(key, value)

                # Log main metric
                mlflow.log_metric(metric_name, float(metric_value))

                # Log additional metrics if provided
                if additional_metrics:
                    mlflow.log_metrics(additional_metrics)

                # Handle signature and input example
                signature = None
                input_example = None
                if X_train is not None:
                    input_example = X_train.iloc[0:1] if hasattr(X_train, 'iloc') else X_train[0:1]
                    
                    if y_train is None:
                        predictions = model.predict(input_example)
                        signature = infer_signature(input_example, predictions)
                    else:
                        signature = infer_signature(input_example, y_train.iloc[0:1] if hasattr(y_train, 'iloc') else y_train[0:1])

                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=self.model_name
                )

                # Log environment info
                mlflow.log_param("python_version", platform.python_version())
                mlflow.log_param("os", platform.system())

                # Get and save best model
                self.get_best_model_run_id(experiment_name=self.experiment_name, metric_name=metric_name)
                self.save_model(dst_path=dst_path, artifact_name=model_name)

                # Generate report
                best_model_name, best_model_parameters, best_model_metrics, report = self.create_run_report()
                
                # Save report as artifact
                mlflow.log_dict(report, "run_report.json")

                return self.best_model_run_id

        except Exception as e:
            logging.error(f"Error in run_mlflow_experiment: {str(e)}")
            raise CustomException(e, sys) from e