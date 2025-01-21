import pandas as pd
from configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME, COLLECTION_NAME

def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Rows and Columns : {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {e}")

def prepare_data_for_mongodb(df):
    """Converts a DataFrame to a list of dictionaries for MongoDB insertion."""
    try:
        df.reset_index(drop=True, inplace=True)
        data_dict = df.to_dict("records")  # Convert DataFrame to list of dictionaries
        if data_dict:
            print(data_dict[0])  # Print the first record to verify
        return data_dict
    except Exception as e:
        raise Exception(f"Error processing the data: {e}")

def insert_data_to_mongodb(database_name, collection_name, data_dict):
    """Inserts data into MongoDB."""
    try:
        mongo_client = MongoDBClient(database_name)  # Create an instance of MongoDBClient
        db = mongo_client.database  # Access the database
        collection = db[collection_name]  # Access the collection
        collection.insert_many(data_dict)  # Insert all records at once
        print(f"Data successfully inserted into {database_name}.{collection_name}")
    except Exception as e:
        raise Exception(f"An error occurred during data insertion: {e}")
    finally:
        # Close MongoDB connection
        mongo_client.close()

def main(file_path, database_name, collection_name):
    """Main function to execute the data dump process."""
    try:
        # Step 1: Read CSV file
        df = read_csv_file(file_path)

        # Step 2: Prepare data for MongoDB
        data_dict = prepare_data_for_mongodb(df)

        # Step 3: Insert data into MongoDB
        insert_data_to_mongodb(database_name, collection_name, data_dict)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    # Define the file path, database, and collection
    DATA_FILE_PATH = (r"E:\ml_project_with_evidentlyAI\data\EasyVisa.csv")
    DATABASE_NAME = DATABASE_NAME
    COLLECTION_NAME = COLLECTION_NAME

    main(DATA_FILE_PATH, DATABASE_NAME, COLLECTION_NAME)
