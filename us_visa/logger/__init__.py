
import os
import logging
from datetime import datetime

# Define log file and log directory
log_dir = 'logs'
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(log_dir, log_file)

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    filename=logs_path,  # Log file location
    level=logging.INFO,  # Log level
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",  # Log format
)

