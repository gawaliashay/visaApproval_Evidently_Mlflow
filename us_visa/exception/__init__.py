import sys
import logging

# Set up a logger for the exception module
logger = logging.getLogger(__name__)

class CustomException(Exception):

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        # Call the error_message_detail method to get a formatted error message
        self.error_message = CustomException.error_message_detail(error_message, error_detail=error_detail)
        
        # Log the exception with the formatted error message
        logger.error(self.error_message)

    @staticmethod
    def error_message_detail(error: Exception, error_detail: sys) -> str:
        """
        error: Exception object raised from a module.
        error_detail: sys module contains detailed information about system execution.
        """
        # Extracting traceback information
        exc_type, exc_value, exc_tb = error_detail.exc_info()

        if exc_tb is None:
            # If no traceback is available, return a default error message
            return f"Error: {str(error)} (No traceback available)"
        
        # Extracting line number and file name from the traceback
        line_number = exc_tb.tb_frame.f_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename

        # Preparing the error message with the traceback details
        error_message = f"Error occurred in Python script [{file_name}] " \
                        f"at line number [{line_number}] with error message [{str(error)}]."

        return error_message

    def __str__(self):
        """
        Formatting how an object should appear when used in a print statement.
        """
        return self.error_message

    def __repr__(self):
        """
        Formatting object representation for debugging purposes.
        """
        return f"{self.__class__.__name__}({repr(self.error_message)})"
