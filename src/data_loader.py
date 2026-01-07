import pandas as pd
import logging
import os

# Configure logger
logger = logging.getLogger(__name__)

def load_and_clean_data(filename):
    """
    Loads the dataset from the 'data' directory and performs initial cleaning.
    """
    try:
        # 1. Get the directory where this script is located (the 'src' folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to the project root directory
        project_root = os.path.dirname(current_dir)
        
        # 3. Construct the full path to the data file inside the 'data' folder
        file_path = os.path.join(project_root, 'data', filename)

        logger.info(f"Looking for file at: {file_path}")

        # Check if the file exists before attempting to read it
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Define columns to drop (technical identifiers not needed for analysis)
        cols_to_drop = ['PatientID', 'DoctorInCharge']
        
        # Check which columns actually exist in the dataframe before dropping
        existing_cols = [c for c in cols_to_drop if c in df.columns]

        if existing_cols:
            df_cleaned = df.drop(columns=existing_cols)
            logger.info(f"Dropped columns: {existing_cols}")
        else:
            df_cleaned = df
        
        logger.info(f"Data loaded successfully. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise