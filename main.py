import logging
from src.data_loader import load_and_clean_data

# 1. Logger Setup
# This configures how messages are displayed (Time - Level - Message)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function.
    Orchestrates the data loading and analysis process.
    """
    logger.info("Starting the project pipeline...")

    try:
        # Step 1: Load Data
        # We call the function we created in src/data_loader.py
        filename = 'alzheimers_disease_data.csv'
        df = load_and_clean_data(filename)
        
        # Temporary print to show it worked (logging is preferred usually)
        print("\n--- Success! Data Sample: ---")
        print(df.head())
        print("-----------------------------\n")

    except Exception as e:
        logger.critical(f"The pipeline crashed: {e}")

if __name__ == "__main__":
    main()