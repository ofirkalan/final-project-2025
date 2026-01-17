import sys
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

def evaluate_model(model, X_test, y_test, scaler, model_name):
    """
    Evaluates model performance including Recall as per project requirements.
    """    
    if model is None:
        logger.warning(f"{model_name} is None. Skipping evaluation.")
        return

    X_test_input = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_test_input)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    # Adding recall calculation to satisfy Success Criteria
    recall = recall_score(y_test, y_pred) 
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"--- {model_name} Performance ---")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Recall: {recall:.4f}") # Critical for the primary research question
    
    plot_confusion_matrix(cm, model_name)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
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
    logger.error(f"CRITICAL ERROR: Missing modules. Details: {e}")
    sys.exit(1)

def evaluate_model(model, X_test, y_test, scaler, model_name):
    if model is None:
        logger.warning(f"{model_name} is None. Skipping evaluation.")
        return

    X_test_input = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_test_input)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"--- {model_name} Performance ---")
    logger.info(f"Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, model_name)

if __name__ == "__main__":
    logger.info("Starting Alzheimer's Analysis System...")

    try:
        df = df_cleaned if 'df_cleaned' in globals() and df_cleaned is not None else load_data()
    except NameError:
        df = load_data()

    if df is not None:
        plot_general_correlation_matrix(df)
        logger.info("Performing Feature Engineering...")
        df = perform_feature_engineering(df).dropna()
        analyze_correlations(df)

        logger.info("Splitting Data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        feature_names = X_train.columns

        logger.info("--- Training Logistic Regression ---")
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        logger.info("--- Training Random Forest ---")
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        logger.info("--- Generating Feature Importance Insights ---")
        if rf_model is not None:
            plot_feature_importance(rf_model, feature_names)

        logger.info("Analysis Complete.")
    else:
        logger.error("No data available. Exiting.")