import sys
import pandas as pd
import numpy as np
import logging # Required for best practices as per guidelines
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure logging settings to replace print statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- IMPORT BLOCK ---
# Attempt to import custom project modules with error handling
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
    )
    
except ImportError as e:
    # Log critical import errors and terminate execution
    logger.error(f"CRITICAL ERROR: Missing modules. Details: {e}")
    sys.exit(1)

def evaluate_model(model, X_test, y_test, scaler, model_name):
    """
    Helper function to evaluate a trained model.
    Logs accuracy and displays a confusion matrix.
    """    
    # Check if the model exists before proceeding
    if model is None:
        logger.warning(f"{model_name} is None. Skipping evaluation.")
        return

    # Apply scaling to test data if required by the model
    if scaler:
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    # Predict and calculate performance metrics
    y_pred = model.predict(X_test_input)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Log metrics to the console
    logger.info(f"--- {model_name} Performance ---")
    logger.info(f"Accuracy: {acc:.4f}")
    
    # Call visualization function
    plot_confusion_matrix(cm, model_name)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    logger.info("Starting Alzheimer's Analysis System...")

    # Load the cleaned dataset
    try:
        if 'df_cleaned' in globals() and df_cleaned is not None:
            df = df_cleaned
        else:
            df = load_data()
    except NameError:
        df = load_data()

    if df is not None:
        # Step 1: Initial data visualization
        plot_general_correlation_matrix(df)

        # Step 2: Process and clean data
        logger.info("Performing Feature Engineering...")
        df = perform_feature_engineering(df)
        df = df.dropna()

        # Step 3: Run statistical significance tests
        analyze_correlations(df)

        # Step 4: Split data for training and testing
        logger.info("Splitting Data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        feature_names = X_train.columns

        # Step 5: Model Training - Logistic Regression
        logger.info("--- Training Logistic Regression ---")
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        # Step 6: Model Training - Random Forest
        logger.info("--- Training Random Forest ---")
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        # Step 7: Visualize which features influenced the decision most
        logger.info("--- Generating Feature Importance Insights ---")
        if rf_model is not None:
            plot_feature_importance(rf_model, feature_names)

        logger.info("Analysis Complete.")
        
    else:
        # Handle cases where data loading failed
        logger.error("No data available. Exiting.")