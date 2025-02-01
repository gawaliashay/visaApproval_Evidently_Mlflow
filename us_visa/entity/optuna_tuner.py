import optuna
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

from typing import Any
import yaml
import logging
import sys
from functools import partial

from us_visa.exception import CustomException

# Read model.yaml configuration
def load_model_config(yaml_path: str) -> dict:
    try:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise CustomException(f"Error reading YAML config: {str(e)}", sys)

# Define objective function for Optuna optimization
def objective(trial: optuna.Trial, config: dict, X_train: Any, y_train: Any) -> float:
    model_selection = config['model_selection']
    model_name = trial.suggest_categorical("model", list(model_selection.keys()))

    model_info = model_selection[model_name]
    model_class = eval(model_info['class'])
    model_params = model_info['params'].copy()

    # Hyperparameters for tuning
    search_param_grid = model_info['search_param_grid']
    for param, values in search_param_grid.items():
        model_params[param] = trial.suggest_categorical(param, values)

    # Instantiate model with tuned hyperparameters
    model = model_class(**model_params)
    
    # Cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
    return score

# Perform Optuna hyperparameter tuning
def tune_model(X_train: Any, y_train: Any, model_config_path: str) -> Any:
    try:
        config = load_model_config(model_config_path)

        # Use functools.partial to bind X_train and y_train to the objective function
        partial_objective = partial(objective, config=config, X_train=X_train, y_train=y_train)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(partial_objective, n_trials=10)
        
        best_trial = study.best_trial
        best_model_name = best_trial.params['model']
        best_params = {param: best_trial.params[param] for param in best_trial.params if param != 'model'}
        
        # Get the best model from the selected model type
        best_model_info = config['model_selection'][best_model_name]
        best_model_class = eval(best_model_info['class'])
        best_model = best_model_class(**best_params)

        # Fit the best model to the training data
        best_model.fit(X_train, y_train)
        
        return best_model, best_params, best_trial.value
    
    except Exception as e:
        raise CustomException(f"Error during Optuna tuning: {str(e)}", sys)
