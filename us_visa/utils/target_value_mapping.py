from us_visa.exception import CustomException
from us_visa.logger import logging
import pandas as pd
import numpy as np

class TargetValueMapping:
    def __init__(self):
        logging.info("Initializing TargetValueMapping class.")
        try:
            self.forward_mapping = {
                "Certified": 0,
                "Denied": 1
            }
            self.reverse_mapping = {v: k for k, v in self.forward_mapping.items()}
        except Exception as e:
            logging.error(f"Error initializing mappings: {e}")
            raise CustomException("Failed to initialize TargetValueMapping.") from e

    def encode(self, target: pd.Series) -> np.ndarray:
        """Encodes a pandas Series using the forward mapping."""
        try:
            encoded_target = target.map(self.forward_mapping).values
            return encoded_target
        except KeyError as e:
            logging.error(f"Target value not found in mapping: {e}")
            raise CustomException(f"Target value not found in mapping: {e}") from e
        except Exception as e:
            logging.error(f"Error encoding target: {e}")
            raise CustomException("Failed to encode target.") from e

    def decode(self, encoded_target: np.ndarray) -> pd.Series:
        """Decodes a numpy array using the reverse mapping."""
        try:
            decoded_target = pd.Series(encoded_target).map(self.reverse_mapping)
            return decoded_target
        except KeyError as e:
            logging.error(f"Encoded value not found in reverse mapping: {e}")
            raise CustomException(f"Encoded value not found in reverse mapping: {e}") from e
        except Exception as e:
            logging.error(f"Error decoding target: {e}")
            raise CustomException("Failed to decode target.") from e

    def as_dict(self):
        logging.info("Returning forward mapping as dictionary.")
        return self.forward_mapping

    def as_reverse_dict(self):
        logging.info("Returning reverse mapping as dictionary.")
        return self.reverse_mapping