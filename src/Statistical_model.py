import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def perform_feature_engineering(df):
    """
    Adds features to the dataframe.
    """
    try:
        df = df.copy()
        
        # Safe binning of Age
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[60, 70, 80, 90, 100], labels=[0, 1, 2, 3])

        # Safe One-Hot Encoding
        if 'Ethnicity' in df.columns:
            df = pd.get_dummies(df, columns=['Ethnicity'], drop_first=True)
            
        return df
    except Exception as e:
        raise RuntimeError(f"[Error] Feature engineering failed: {e}") from e

def preprocess_data(df):
    """
    Splits data into Train/Test.
    """
    try:
        # filtering columns (ignoring errors if columns don't exist)
        X = df.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1, errors='ignore')
        y = df['Diagnosis']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        raise RuntimeError(f"[Error] Data preprocessing failed: {e}") from e

def train_logistic_regression(X_train, y_train):
    """
    Trains Logistic Regression with error handling.
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)

        return model, scaler
    except Exception as e:
        raise RuntimeError(f"[Error] Logistic Regression training failed: {e}") from e

def train_random_forest(X_train, y_train):
    """
    Trains Random Forest with error handling.
    """
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise RuntimeError(f"[Error] Random Forest training failed: {e}") from e