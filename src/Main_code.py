import sys
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- IMPORT BLOCK ---
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
    logging.critical(f"CRITICAL ERROR: Missing modules. Details: {e}")
    sys.exit(1)

def evaluate_model(model, X_test, y_test, scaler, model_name):
    """
    Helper function to evaluate a trained model.
    Calculates Accuracy AND Recall, then plots confusion matrix.
    """    
    if model is None:
        logging.warning(f"Warning: {model_name} is None. Skipping evaluation.")
        return

    if scaler:
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    # Generate predictions
    y_pred = model.predict(X_test_input)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Log results
    logging.info(f"--- {model_name} Performance ---")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Recall:   {recall:.4f}")
    
    # Check if Recall meets the success criteria (> 80%)
    if recall < 0.80:
        logging.warning(f"(!) Recall is below 0.80. Consider tuning class_weight or threshold.")
    else:
        logging.info(f"(v) Recall meets the success criteria (>0.80).")

    # Plotting
    plot_confusion_matrix(cm, model_name)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    logging.info("Starting Alzheimer's Analysis System...")

    try:
        if 'df_cleaned' in globals() and df_cleaned is not None:
            df = df_cleaned
            logging.info("Data loaded from global context.")
        else:
            df = load_data()
            logging.info("Data loaded via load_data() function.")
    except NameError:
        df = load_data()

    if df is not None:
        logging.info("Generating general correlation matrix...")
        plot_general_correlation_matrix(df)

        logging.info("Performing Feature Engineering...")
        df = perform_feature_engineering(df)
        
        # --- Verification Check ---
        if 'CardioRiskScore' in df.columns:
            logging.info("✅ 'CardioRiskScore' created successfully!")
            logging.info(f"Sample values (First 5): {df['CardioRiskScore'].head().tolist()}")
        else:
            logging.warning("⚠️ 'CardioRiskScore' was NOT created.")
        # --------------------------

        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
             logging.info(f"Dropped {original_len - len(df)} rows containing NaN values.")

        logging.info("Running statistical analysis...")
        analyze_correlations(df)

        logging.info("Splitting Data into Train/Test sets...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        feature_names = X_train.columns

        logging.info("Training Logistic Regression model...")
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        logging.info("Training Random Forest model...")
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        logging.info("Generating Feature Importance Insights...")
        if rf_model is not None:
            plot_feature_importance(rf_model, feature_names)

        logging.info("Analysis Complete. Exiting successfully.")
        
    else:
        logging.error("No data available. Process cannot continue.")