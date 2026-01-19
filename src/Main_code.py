import sys
import pandas as pd
import numpy as np
import logging  # Imported logging module

# --- LOGGING SETUP ---
# Configuring the logger to show the time, level (INFO/ERROR), and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- IMPORT BLOCK ---
# We use a try-except block to safely import our custom modules.
# If a file is missing or has an error, the code will stop here and tell us why.
try:
    # 1. Load Data Module
    from Text_read import load_data, df_cleaned
    
    # 2. Statistics Module
    from Statistic_analysis import analyze_correlations
    
    # 3. Visualization Module
    # We import ALL 3 plotting functions here.
    from Main_plots import (
        plot_confusion_matrix, 
        plot_feature_importance, 
        plot_general_correlation_matrix
    )
    
    # 4. Modeling Module (Preprocessing & Training)
    # Added evaluate_model to the import list
    from Statistical_model import (
        perform_feature_engineering,
        preprocess_data,
        train_logistic_regression,
        train_random_forest,
        evaluate_model 
    )
    
except ImportError as e:
    # Using critical for errors that stop the program
    logging.critical(f"CRITICAL ERROR: Missing modules. Details: {e}")
    sys.exit(1) # Stop the program immediately

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    logging.info("Starting Alzheimer's Analysis System...")

    # Step 1: Load the Data
    # We check if 'df_cleaned' is already available (from import) or needs to be loaded.
    try:
        if 'df_cleaned' in globals() and df_cleaned is not None:
            df = df_cleaned
            logging.info("Data loaded from global context.")
        else:
            df = load_data()
            logging.info("Data loaded via load_data() function.")
    except NameError:
        df = load_data()

    # Only proceed if data was loaded successfully
    if df is not None:
        
        # --- NEW STEP: Visualizing Raw Data ---
        # We plot the general correlation matrix BEFORE making any changes to the data.
        # This gives us an initial look at the relationships.
        logging.info("Generating general correlation matrix...")
        plot_general_correlation_matrix(df)

        # Step 2: Feature Engineering
        logging.info("Performing Feature Engineering...")
        df = perform_feature_engineering(df)
        
        # Remove rows with NaN values to prevent model crashes
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
             logging.info(f"Dropped {original_len - len(df)} rows containing NaN values.")

        # Step 3: Statistical Analysis (P-Value check)
        logging.info("Running statistical analysis...")
        analyze_correlations(df)

        # Step 4: Split Data into Train and Test sets
        logging.info("Splitting Data into Train/Test sets...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # We save the column names now, to use them later in the Feature Importance graph
        feature_names = X_train.columns

        # Step 5: Train & Evaluate Logistic Regression
        logging.info("Training Logistic Regression model...")
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        # Step 6: Train & Evaluate Random Forest
        logging.info("Training Random Forest model...")
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        # Step 7: Final Insights
        logging.info("Generating Feature Importance Insights...")
        if rf_model is not None:
            plot_feature_importance(rf_model, feature_names)

        logging.info("Analysis Complete. Exiting successfully.")
        
    else:
        logging.error("No data available. Process cannot continue.")