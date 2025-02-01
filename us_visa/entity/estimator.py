import numpy as np
import logging
import sys
from us_visa.exception import CustomException  # Update with correct import

class USvisaModel:
    def __init__(self, trained_model_object, preprocessing_object):
        self.trained_model_object = trained_model_object
        self.preprocessing_object = preprocessing_object

    def predict(self, array: np.ndarray) -> np.ndarray:
        """
        Accepts raw inputs (array), transforms them using preprocessing_object,
        and then performs prediction using the trained model.
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            # Log the shape of the input data
            logging.info(f"Original input shape: {array.shape}")

            # If the array is 2D, we can assume the last column is the target, and the rest are features
            if len(array.shape) > 1 and array.shape[1] > 1:
                logging.info(f"Input has {array.shape[1]} features.")

            # Transform the raw input features using preprocessing_object
            transformed_feature = self.preprocessing_object.transform(array)

            # Perform prediction
            predictions = self.trained_model_object.predict(transformed_feature)

            # Return the predictions as a numpy array
            return predictions

        except Exception as e:
            # Log the error and raise a custom exception
            logging.error(f"Error in prediction: {e}")
            raise CustomException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
