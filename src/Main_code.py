import sys
import pandas as pd
import numpy as np

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
    from Statistical_model import (
        perform_feature_engineering,
        preprocess_data,
        train_logistic_regression,
        train_random_forest
    )
    
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. Details: {e}")
    sys.exit(1) # Stop the program immediately

def evaluate_model(model, X_test, y_test, scaler, model_name):
    """
    Helper function to evaluate a trained model.
    It calculates accuracy and calls the plot_confusion_matrix function.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # Scaling Logic:
    # If the model used a scaler (like Logistic Regression), we must scale the test data too.
    # If not (like Random Forest), we use the data as is.
    if scaler:
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    # Generate predictions using the model
    y_pred = model.predict(X_test_input)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {acc:.4f}")
    
    # Call the plotting function from Main_plots.py
    plot_confusion_matrix(cm, model_name)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("🚀 Starting Alzheimer's Analysis System...")

    # Step 1: Load the Data
    # We check if 'df_cleaned' is already available (from import) or needs to be loaded.
    try:
        if 'df_cleaned' in globals() and df_cleaned is not None:
            df = df_cleaned
        else:
            df = load_data()
    except NameError:
        df = load_data()

    # Only proceed if data was loaded successfully
    if df is not None:
        
        # --- NEW STEP: Visualizing Raw Data ---
        # We plot the general correlation matrix BEFORE making any changes to the data.
        # This gives us an initial look at the relationships.
        plot_general_correlation_matrix(df)

        # Step 2: Feature Engineering
        print("Performing Feature Engineering...")
        df = perform_feature_engineering(df)
        
        # Step 3: Statistical Analysis (P-Value check)
        analyze_correlations(df)

        # Step 4: Split Data into Train and Test sets
        print("Splitting Data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # We save the column names now, to use them later in the Feature Importance graph
        feature_names = X_train.columns

        # Step 5: Train & Evaluate Logistic Regression
        print("\n--- Training Logistic Regression ---")
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        # Step 6: Train & Evaluate Random Forest
        print("\n--- Training Random Forest ---")
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        # Step 7: Final Insights
        print("\n--- Generating Feature Importance Insights ---")
        plot_feature_importance(rf_model, feature_names)

        print("\n Analysis Complete.")
        
    else:
        print(" No data available. Exiting.")