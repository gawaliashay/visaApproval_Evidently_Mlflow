
# Importing the required libraries
from us_visa.pipeline.training_pipeline import TrainPipeline
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the number of cores you want to use

# Creating an object of the TrainPipeline class
obj = TrainPipeline()
obj.run_pipeline()