import unittest
import pandas as pd
import numpy as np
from Analysis_main import (
    perform_feature_engineering, 
    preprocess_data, 
    train_logistic_regression, 
    train_random_forest, 
    evaluate_model, 
    get_feature_importance,
    analyze_correlations
)

class TestAlzheimersProject(unittest.TestCase):
    
    def setUp(self):
        """
        Setup: This runs before EVERY test.
        We create a fake DataFrame that looks exactly like the real data,
        so we can test the functions without needing the real CSV file.
        """
#creating fake data 
        self.data = pd.DataFrame({
            'Age': [60, 70, 80, 65, 75, 85, 62, 72, 82, 68],
            'Hypertension': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Diabetes': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            'Smoking': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'Ethnicity': [0, 1, 2, 0, 3, 0, 1, 2, 0, 1], # For One-Hot Encoding
            'PhysicalActivity': [5, 2, 8, 4, 6, 1, 9, 3, 7, 5], # For stats
            'DietQuality': [7, 4, 9, 5, 8, 3, 9, 4, 8, 6],
            'SleepQuality': [8, 5, 7, 6, 8, 4, 9, 5, 7, 6],
            'MMSE': [25, 18, 10, 28, 15, 8, 29, 20, 12, 26],
            'Diagnosis': [0, 1, 1, 0, 1, 1, 0, 0, 1, 0] # Target
        })

    # --- Test 1: Feature Engineering ---
    def test_perform_feature_engineering(self):
        """
        Test that new features (CardioRisk, AgeGroup, One-Hot Ethnicity) are created correctly.
        """
        # Act: activating the function
        df_eng = perform_feature_engineering(self.data)
        
        # Assert: checking
        # 1. Check if 'CardioRiskScore' was created
        self.assertIn('CardioRiskScore', df_eng.columns)
        # Check calculation: row 1 (index 1): Hyp(1) + Dia(0) + Smo(1) = 2
        self.assertEqual(df_eng.iloc[1]['CardioRiskScore'], 2)
        
        # 2. Check if 'AgeGroup' was created
        self.assertIn('AgeGroup', df_eng.columns)
        
        # 3. Check One-Hot Encoding (Ethnicity column should be gone, Ethnicity_1 etc. added)
        self.assertNotIn('Ethnicity', df_eng.columns)
        self.assertTrue(any(col.startswith('Ethnicity_') for col in df_eng.columns))

    # --- Test 2: Statistical Analysis ---
    def test_analyze_correlations(self):
        """
        Test that the statistical function runs and returns 3 results (Activity, Diet, Sleep).
        """
        results = analyze_correlations(self.data)
        
        # We expect 3 tuples in the results list (one for each lifestyle factor)
        self.assertEqual(len(results), 3)
        # Check that the first item contains 'PhysicalActivity'
        self.assertEqual(results[0][0], 'PhysicalActivity')

    # --- Test 3: Preprocessing (Splitting) ---
    def test_preprocess_data(self):
        """
        Test splitting into Train/Test and X/y.
        """
        # First we need to engineer the data because preprocess expects the new columns
        df_eng = perform_feature_engineering(self.data)
        
        X_train, X_test, y_train, y_test = preprocess_data(df_eng, target_column='Diagnosis')
        
        # Check split ratio (80% train = 8 rows, 20% test = 2 rows)
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        
        # Check that Diagnosis is NOT in X (features)
        self.assertNotIn('Diagnosis', X_train.columns)

    # --- Test 4: Logistic Regression Training ---
    def test_train_logistic_regression(self):
        """
        Test that Logistic Regression trains and returns a model and scaler.
        """
        # Prepare data
        df_eng = perform_feature_engineering(self.data)
        X_train, X_test, y_train, y_test = preprocess_data(df_eng)
        
        # Train
        model, scaler = train_logistic_regression(X_train, y_train)
        
        # Assert
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        # Check if model has coefficients (means it is trained)
        self.assertTrue(hasattr(model, 'coef_'))

    # --- Test 5: Random Forest Training ---
    def test_train_random_forest(self):
        """
        Test that Random Forest trains correctly.
        """
        # Prepare data
        df_eng = perform_feature_engineering(self.data)
        X_train, X_test, y_train, y_test = preprocess_data(df_eng)
        
        # Train
        model = train_random_forest(X_train, y_train)
        
        # Assert
        self.assertIsNotNone(model)
        # Random Forest has 'feature_importances_', not 'coef_'
        self.assertTrue(hasattr(model, 'feature_importances_'))

   # --- Test 6: Model Evaluation ---
    def test_evaluate_model(self):
        """
        Test that evaluation returns Accuracy and Recall scores.
        """
        # Setup
        df_eng = perform_feature_engineering(self.data)
        X_train, X_test, y_train, y_test = preprocess_data(df_eng)
        model = train_random_forest(X_train, y_train)
        
        # Act
        accuracy, recall = evaluate_model(model, X_test, y_test, model_name="TestRF")
        
        # Assert (Scores must be between 0 and 1)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(0 <= recall <= 1)

    # --- Test 7: Feature Importance ---
    def test_get_feature_importance(self):
        """
        Test that we get a sorted table of important features.
        """
        # Setup
        df_eng = perform_feature_engineering(self.data)
        X_train, X_test, y_train, y_test = preprocess_data(df_eng)
        model = train_random_forest(X_train, y_train)
        
        # Act
        imp_df = get_feature_importance(model, X_train.columns, model_type='tree')
        
        # Assert
        self.assertIn('Feature', imp_df.columns)
        self.assertIn('Importance', imp_df.columns)
        # Check sorting: First item should have >= importance than last item
        self.assertGreaterEqual(imp_df.iloc[0]['Abs_Importance'], imp_df.iloc[-1]['Abs_Importance'])

if __name__ == '__main__':
    unittest.main()