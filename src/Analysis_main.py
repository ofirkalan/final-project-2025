import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats  # For statistical significance (p-value)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

# --- 1. Data Loading ---

def load_data():
    """Tries to import data from Text_read.py"""
    try:
        from Text_read import df_cleaned
        print("\n--- Successfully imported 'df_cleaned' from Text_read.py ---")
        return df_cleaned.copy() # Return a copy to avoid warnings
    except ImportError:
        print("Error: Could not import 'Text_read.py'. Make sure both files are in the same folder.")
        return None

# --- 2. Feature Engineering & Preprocessing ---

def perform_feature_engineering(df):
    """
    Adds new features based on the requirements:
    1. Cardio Risk Score
    2. Age Groups
    3. One-Hot Encoding for Ethnicity
    """
    df_eng = df.copy()

    # Requirement: Create 'Cardiovascular Risk Score'
    # Combining Hypertension, Diabetes, and Smoking (simple sum)
    df_eng['CardioRiskScore'] = df_eng['Hypertension'] + df_eng['Diabetes'] + df_eng['Smoking']

    # Requirement: Age Binning (Groups)
    # 0=60-69, 1=70-79, 2=80+
    df_eng['AgeGroup'] = pd.cut(df_eng['Age'], bins=[59, 69, 79, 100], labels=[0, 1, 2])

    # Requirement: One-Hot Encoding for Ethnicity (Nominal variable)
    # This turns 'Ethnicity' column into 'Ethnicity_0', 'Ethnicity_1', etc.
    df_eng = pd.get_dummies(df_eng, columns=['Ethnicity'], prefix='Ethnicity', drop_first=True)
    
    # Ensuring AgeGroup is also treated properly
    df_eng['AgeGroup'] = df_eng['AgeGroup'].astype(int)

    return df_eng

def preprocess_data(df, target_column='Diagnosis'):
    """
    Splits features (X) and target (y), and performs train-test split.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Stratify ensures we maintain the same proportion of sick/healthy people in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

# --- 3. Statistical Analysis ---

def analyze_correlations(df):
    """
    Requirement: Check correlation and p-value between Lifestyle factors and MMSE.
    """
    print("\n--- Statistical Analysis (Lifestyle vs MMSE) ---")
    lifestyle_factors = ['PhysicalActivity', 'DietQuality', 'SleepQuality']
    target = 'MMSE'
    
    results = []
    for factor in lifestyle_factors:
        # Calculate Pearson correlation and p-value
        corr, p_value = stats.pearsonr(df[factor], df[target])
        results.append((factor, corr, p_value))
        print(f"{factor} vs {target}: Correlation={corr:.3f}, p-value={p_value:.5f}")
        
        if p_value < 0.05:
            print(f"  -> Significant relationship found for {factor}!")
    
    return results

# --- 4. Modeling ---

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- 5. Evaluation ---

def evaluate_model(model, X_test, y_test, scaler=None, model_name="Model"):
    """
    Evaluates model performance with focus on Recall (Sensitivity).
    """
    if scaler:
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    y_pred = model.predict(X_test_input)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred) # Important requirement!
    
    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall (Sensitivity): {recall:.2%}") # Checking against the >80% criteria
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, recall

def get_feature_importance(model, feature_names, model_type='linear'):
    """
    Extracts feature importance based on model type.
    """
    if model_type == 'linear':
        importance = model.coef_[0]
    else: # Random Forest
        importance = model.feature_importances_

    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    feature_imp['Abs_Importance'] = feature_imp['Importance'].abs()
    feature_imp = feature_imp.sort_values(by='Abs_Importance', ascending=False)
    
    return feature_imp

# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Load Data
    df = load_data()
    
    if df is not None:
        # --- Graph 1: Correlation Heatmap (Raw Data) ---
        plt.figure(figsize=(12, 10))
        # Select only numeric columns for correlation to avoid errors
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Matrix (Raw Data)")
        plt.show()

        # 2. Statistical Analysis
        analyze_correlations(df)
        
        # 3. Feature Engineering
        df_engineered = perform_feature_engineering(df)
        print(f"\nData Shape after Engineering: {df_engineered.shape}")

        # 4. Preprocessing
        X_train, X_test, y_train, y_test = preprocess_data(df_engineered)

        # 5. Train & Evaluate Logistic Regression
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_test, y_test, scaler, "Logistic Regression")

        # 6. Train & Evaluate Random Forest (Requirement)
        rf_model = train_random_forest(X_train, y_train)
        evaluate_model(rf_model, X_test, y_test, None, "Random Forest")

        # --- Graph 2: Confusion Matrix for Random Forest ---
        y_pred_rf = rf_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_rf)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Random Forest')
        plt.show()

        # 7. Feature Importance
        top_features = get_feature_importance(rf_model, X_train.columns, model_type='tree')
        print("\nTop 5 Risk Factors (Random Forest):")
        print(top_features[['Feature', 'Importance']].head(5))

        # --- Graph 3: Feature Importance Bar Plot ---
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features.head(10), palette='viridis')
        plt.title('Top 10 Most Important Features for Predicting Alzheimer\'s')
        plt.xlabel('Importance Score')
        plt.show()