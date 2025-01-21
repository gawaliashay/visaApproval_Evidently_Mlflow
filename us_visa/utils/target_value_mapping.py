from us_visa.exception import USvisaException
from us_visa.logger import logging

class TargetValueMapping:
    def __init__(self):
        logging.info("Initializing TargetValueMapping class.")
        
        try:
            # Define forward and reverse mappings during initialization
            self.forward_mapping = {
                "Certified": 0,
                "Denied": 1
            }
            self.reverse_mapping = {v: k for k, v in self.forward_mapping.items()}
        except Exception as e:
            logging.error(f"Error initializing mappings: {e}")
            raise USvisaException("Failed to initialize TargetValueMapping.") from e

    def as_dict(self):
        logging.info("Returning forward mapping as dictionary.")
        return self.forward_mapping

    def as_reverse_dict(self):
        logging.info("Returning reverse mapping as dictionary.")
        return self.reverse_mapping
