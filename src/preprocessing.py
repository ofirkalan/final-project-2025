import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df, target_column='Diagnosis'):
    """
    Performs Feature Engineering and Preprocessing tailored to the Research Questions.
    
    Research Question 1 Strategy:
    Predict Diagnosis based ONLY on modifiable lifestyle factors and clinical metrics.
    We EXCLUDE:
    1. Non-modifiable demographics (Age, Gender, Ethnicity, Family History).
    2. Outcomes/Symptoms (MMSE, ADL, Confusion, etc.) to prevent data leakage.
    """
    try:
        logger.info("Starting preprocessing for Research Question 1...")
        
        df_processed = df.copy()

        # --- 1. Feature Engineering: Risk Scores ---
        # יצירת ציון סיכון בריאותי (מבוסס על גורמים ניתנים לשינוי/טיפול)
        df_processed['HealthRiskScore'] = 0
        risk_factors = ['Diabetes', 'Hypertension', 'Smoking', 'CardiovascularDisease']
        for col in risk_factors:
            if col in df_processed.columns:
                df_processed['HealthRiskScore'] += df_processed[col]
        
        if 'BMI' in df_processed.columns:
            df_processed['HealthRiskScore'] += (df_processed['BMI'] > 30).astype(int)

        # --- 2. Defining Columns to DROP ---
        
        # א. נתונים דמוגרפיים שאינם ניתנים לשינוי (לפי שאלה ראשית)
        non_modifiable = [
            'Age', 'Gender', 'Ethnicity', 'FamilyHistoryAlzheimers'
        ]

        # ב. תסמינים ומדדים קוגניטיביים (כי הם התוצאה, לא הסיבה - Data Leakage)
        # אנחנו מסירים אותם כדי שהמודל לא "ירמה"
        symptoms_and_outcomes = [
            'MMSE', 'FunctionalAssessment', 'ADL', 
            'MemoryComplaints', 'BehavioralProblems',
            'Confusion', 'Disorientation', 'PersonalityChanges', 
            'DifficultyCompletingTasks', 'Forgetfulness'
        ]

        # ג. מידע טכני
        technical = ['PatientID', 'DoctorInCharge']

        # איחוד כל העמודות למחיקה
        columns_to_remove = non_modifiable + symptoms_and_outcomes + technical
        
        # מחיקה בפועל (רק עמודות שקיימות בדאטה)
        cols_to_drop = [target_column] + [col for col in columns_to_remove if col in df_processed.columns]
        
        logger.info(f"Dropping the following columns to ensure pure prediction: {cols_to_drop}")
        
        X = df_processed.drop(columns=cols_to_drop)
        y = df_processed[target_column]

        # --- 3. Split and Scale ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        feature_names = X.columns.tolist()

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise