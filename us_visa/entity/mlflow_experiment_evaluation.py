import sys
import logging
import os
from typing import Tuple, Dict, Any, Optional
import mlflow
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
            # Fetch the experiment by name
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise CustomException(f"Experiment '{experiment_name}' not found.")

            # Determine the order for sorting
            order = f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"

            # Search for runs in the experiment, ordered by the specified metric
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=[order])
            if runs.empty:
                raise CustomException(f"No runs found for experiment '{experiment_name}'.")

            # Extract the best run
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

        Raises:
            CustomException: If the model cannot be loaded or saved.
        """
        try:
            # Fetch the run details
            run = mlflow.get_run(self.best_model_run_id)
            self.artifact_uri = run.info.artifact_uri

            # Construct the correct model URI
            model_uri = f"{self.artifact_uri}/{artifact_name}"

            # Load the model based on the specified format
            if model_format == "pyfunc":
                model = mlflow.pyfunc.load_model(model_uri)
            elif model_format == "sklearn":
                model = mlflow.sklearn.load_model(model_uri)
            else:
                raise ValueError(f"Unsupported model format: {model_format}")

            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Save the model to the specified path
            save_object(file_path=dst_path, obj=model)
            logging.info(f"Model saved successfully to: {dst_path}")

        except Exception as e:
            logging.error(f"Error in save_model: {str(e)}")
            raise CustomException(e, sys) from e

    def create_run_report(self) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Generate a report for the best model run.

        Returns:
            Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]: A tuple containing:
                - Model name
                - Best model parameters
                - Best model metrics
                - A detailed report dictionary
        """
        try:
            # Fetch the best run details
            best_run = mlflow.get_run(self.best_model_run_id)
            best_model_metrics = best_run.data.metrics
            best_model_parameters = best_run.data.params

            # Create a detailed report
            report = {
                "best_run_id": best_run.info.run_id,
                "best_run_experiment_id": best_run.info.experiment_id,
                "best_run_artifact_uri": best_run.info.artifact_uri,
                "best_model_name": self.model_name,
                "best_model_parameters": best_model_parameters,
                "best_model_metrics": best_model_metrics,
            }

            logging.info(f"Run report generated for model: {self.model_name}")
            return self.model_name, best_model_parameters, best_model_metrics, report

        except Exception as e:
            logging.error(f"Error in create_run_report: {str(e)}")
            raise CustomException(e, sys) from e

    def run_mlflow_experiment(
        self, metric_name: str, metric_value: float, model: Any, best_model_parameters: Dict[str, Any], model_name: str, dst_path: str
    ) -> str:
        """
        Execute an MLflow experiment, log the best model, and generate a report.

        Args:
            metric_name (str): Name of the metric to log.
            metric_value (float): Value of the metric.
            model (Any): The trained model to log.
            best_model_parameters (Dict[str, Any]): Parameters of the best model.
            model_name (str): Name of the model.
            dst_path (str): Destination path to save the model locally.

        Returns:
            str: The run ID of the best model.

        Raises:
            CustomException: If the experiment fails.
        """
        try:
            # Set the MLflow experiment
            mlflow.set_experiment(self.experiment_name)

            # Start an MLflow run
            with mlflow.start_run(run_name=self.run_name):
                # Log model parameters
                for key, value in best_model_parameters.items():
                    mlflow.log_param(key, value)

                # Log the specified metric
                mlflow.log_metric(metric_name, float(metric_value))

                # Log the model
                mlflow.sklearn.log_model(model, artifact_path=model_name)

                # Log experiment details
                logging.info(f"MLflow run completed for model '{model_name}'.")
                logging.info(f"Best model metric > {metric_name}: {metric_value}.")
                logging.info(f"Best model parameters: {best_model_parameters}.")
                logging.info(f"Best model saved at: {model_name}.")

                # Retrieve and save the best model
                self.get_best_model_run_id(experiment_name=self.experiment_name, metric_name=metric_name)

                # Save the model locally using the provided dst_path
                self.save_model(dst_path=dst_path, artifact_name=model_name)

                # Generate a report for the best model
                best_model_name, best_model_parameters, best_model_metrics, report = self.create_run_report()
                logging.info(f"Best model found: {best_model_name}.")
                logging.info(f"Best model Run ID: {self.best_model_run_id}.")

            return self.best_model_run_id

        except Exception as e:
            logging.error(f"Error in run_mlflow_experiment: {str(e)}")
            raise CustomException(e, sys) from e