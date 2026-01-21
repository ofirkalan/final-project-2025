import sys
import pandas as pd
import numpy as np
import logging  

# Configuring the logger to show the time, level (INFO/ERROR), and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Imports from the other moduels
try:
    from Text_read import load_data, df_cleaned
    
    from Statistic_analysis import analyze_correlations
    
    from Main_plots import (
        plot_confusion_matrix, 
        plot_feature_importance, 
        plot_general_correlation_matrix
    )
    
    from Statistical_model import (
        perform_feature_engineering,
        preprocess_data,
        train_logistic_regression,
        train_random_forest,
        evaluate_model 
    )
    
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Missing modules. Details: {e}")
    sys.exit(1) # Stop the program immediately

if __name__ == "__main__":
    logging.info("Starting Alzheimer's Analysis System...")

    #Load the Data
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
        
        #Visualizing Raw Data
        logging.info("Generating general correlation matrix...")
        plot_general_correlation_matrix(df)

        #Feature Engineering
        logging.info("Performing Feature Engineering...")
        df = perform_feature_engineering(df)
        
        # Remove rows with NaN values to prevent model crashes
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
             logging.info(f"Dropped {original_len - len(df)} rows containing NaN values.")

        #Statistical Analysis (P-Value check)
        logging.info("Running statistical analysis...")
        analyze_correlations(df)

        #Split Data into Train and Test sets
        logging.info("Splitting Data into Train/Test sets...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        feature_names = X_train.columns

        #Train & Evaluate Logistic Regression
        logging.info("Training Logistic Regression model...")
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        # Train & Evaluate Random Forest
        logging.info("Training Random Forest model...")
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        #Final Insights
        logging.info("Generating Feature Importance Insights...")
        if rf_model is not None:
            plot_feature_importance(rf_model, feature_names)

        logging.info("Analysis Complete. Exiting successfully.")
        
    else:
        logging.error("No data available. Process cannot continue.")