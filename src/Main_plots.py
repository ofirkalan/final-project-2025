import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

def plot_general_correlation_matrix(df):
    """
    Plots a heatmap showing the correlation between ALL numeric variables in the dataset.
    This helps us understand the raw data before we process it.
    """
    try:
        logger.info("Generating General Correlation Matrix (Raw Data)...")
        
        # Open a new figure window with a specific size (width=12, height=10)
        plt.figure(figsize=(12, 10))
        
        # IMPORTANT: We must select only numeric columns (int, float).
        # Calculating correlation on text columns (strings) will cause the code to crash.
        numeric_df = df.select_dtypes(include=[np.number])
        
        # sns.heatmap: Draws the colored matrix.
        # numeric_df.corr(): Calculates the mathematical relationship (-1 to 1).
        # cmap='coolwarm': Sets colors (Red = Positive correlation, Blue = Negative).
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', linewidths=0.5)
        
        plt.title("Correlation Matrix (All Variables)")
        plt.show() # Display the graph
        
        logger.info("General Correlation Matrix displayed successfully.")

    except Exception as e:
        logger.error(f"Could not plot General Correlation Matrix: {e}")
        raise RuntimeError(f"[Error] Could not plot General Correlation Matrix: {e}") from e

def plot_confusion_matrix(cm, model_name):
    """
    Plots the Confusion Matrix (the blue squares).
    It shows how many patients were correctly diagnosed (True Positives/Negatives)
    vs how many were mistakes (False Positives/Negatives).
    """
    try:
        logger.info(f"Generating Confusion Matrix for {model_name}...")
        plt.figure(figsize=(8, 6))
        
        # annot=True: Writes the actual numbers inside the squares.
        # fmt='d': Formats the numbers as integers (no decimal points).
        # cbar=False: Hides the color bar on the side (cleaner look).
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label (What the model thought)')
        plt.ylabel('True Label (What is actually true)')
        plt.show()
        
        logger.info(f"Confusion Matrix for {model_name} displayed successfully.")
        
    except Exception as e:
        logger.error(f"Could not plot Confusion Matrix: {e}")
        raise RuntimeError(f"[Error] Could not plot Confusion Matrix: {e}") from e

def plot_feature_importance(model, feature_names):
    """
    Plots a bar chart showing which features (columns) were most important
    for the Random Forest model to make its decision.
    """
    try:
        logger.info("Generating Feature Importance Graph...")
        
        # Safety Check: Ensure the model actually has importance data.
        # (Logistic Regression, for example, does not have 'feature_importances_').
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not support feature importance. Skipping plot.")
            return

        # Extract the importance numbers from the trained model
        importances = model.feature_importances_
        
        # Create a temporary DataFrame to associate each score with its feature name.
        # We sort it (sort_values) so the graph looks organized (highest to lowest).
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Draw the bar chart
        plt.figure(figsize=(10, 6))
        # palette='viridis': Sets a nice green-purple color scheme
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
        
        plt.title('Feature Importance Analysis')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()
        
        logger.info("Feature Importance Graph displayed successfully.")

    except Exception as e:
        logger.error(f"Could not plot Feature Importance: {e}")
        raise RuntimeError(f"[Error] Could not plot Feature Importance: {e}") from e