import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from us_visa.exception import CustomException
from us_visa.logger import logging


class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Fitted preprocessing pipeline object (e.g., StandardScaler, OneHotEncoder).
        :param trained_model_object: Trained model object (e.g., RandomForest, LogisticRegression).
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts raw inputs (dataframe), transforms them using preprocessing_object, 
        and then performs prediction using the trained model.
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            # Log the shape of the input data
            logging.info(f"Original input shape: {dataframe.shape}")
            logging.info(f"Original DataFrame columns: {dataframe.columns.tolist()}")

            # Transform the raw input features
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Perform prediction
            predictions = self.trained_model_object.predict(transformed_feature)

            # Return the predictions as a DataFrame
            return pd.DataFrame(predictions, columns=['Prediction'])

        except Exception as e:
            # Log the error and raise a custom exception
            logging.error(f"Error in prediction: {e}")
            raise CustomException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
