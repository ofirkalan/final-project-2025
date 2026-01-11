import unittest
import pandas as pd
import numpy as np

# --- IMPORTS FROM YOUR NEW FILE NAMES ---
# We import the logic functions from 'Statistical_model.py'
from Statistical_model import (
    perform_feature_engineering,
    preprocess_data,
    train_logistic_regression
)

# We import the stats function from 'Statistic_analysis.py'
from Statistic_analysis import analyze_correlations

class TestAlzheimersProject(unittest.TestCase):
    """
    Unit Tests for the Alzheimer's Analysis Project.
    This file checks if the logic in 'Statistical_model.py' works correctly.
    """

    def setUp(self):
        """
        This runs automatically BEFORE every test.
        It creates a small, fake dataset so we don't need the real file.
        """
        data = {
            'Age': [60, 72, 85, 95, 65, 75],
            'Ethnicity': [0, 1, 0, 2, 3, 1],
            'Diagnosis': [0, 1, 1, 0, 0, 1],
            'DietQuality': [5.5, 2.1, 8.0, 4.5, 6.0, 3.0],
            'MMSE': [28, 15, 10, 26, 29, 12],
            'PatientID': [101, 102, 103, 104, 105, 106], # Should be dropped
            'DoctorInCharge': ['Dr.A', 'Dr.B', 'Dr.A', 'Dr.C', 'Dr.B', 'Dr.A'] # Should be dropped
        }
        self.mock_data = pd.DataFrame(data)

    def test_feature_engineering(self):
        """
        Test 1: Check if 'Statistical_model.py' correctly adds new features.
        """
        # Run the function
        df_result = perform_feature_engineering(self.mock_data)

        # Check: Did it create the 'AgeGroup' column?
        self.assertIn('AgeGroup', df_result.columns, "Failed to create 'AgeGroup' column")
        
        # Check: Did it perform One-Hot Encoding on Ethnicity?
        # (Columns should increase because Ethnicity 0,1,2,3 became multiple columns)
        self.assertTrue(len(df_result.columns) > len(self.mock_data.columns), 
                        "One-Hot Encoding did not add new columns")

    def test_data_splitting(self):
        """
        Test 2: Check if 'Statistical_model.py' correctly splits Train/Test.
        """
        # Run the function
        X_train, X_test, y_train, y_test = preprocess_data(self.mock_data)

        # Check: Are the outputs valid?
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_test)

        # Check: Did we lose any data? (Train + Test should equal Total)
        total_rows = len(X_train) + len(X_test)
        self.assertEqual(total_rows, len(self.mock_data), "Data rows were lost during splitting")

        # Check: Did we drop the target 'Diagnosis' from X?
        self.assertNotIn('Diagnosis', X_train.columns, "Target variable 'Diagnosis' leaked into X_train")

    def test_model_training(self):
        """
        Test 3: Check if the model trains without crashing.
        """
        # Prepare data
        X_train, X_test, y_train, y_test = preprocess_data(self.mock_data)

        # Run the training function
        model, scaler = train_logistic_regression(X_train, y_train)

        # Check: Did it return a real model and scaler?
        self.assertIsNotNone(model, "Model training returned None")
        self.assertIsNotNone(scaler, "Scaler returned None")

    def test_statistics_run(self):
        """
        Test 4: Check if 'Statistic_analysis.py' runs without errors.
        """
        # We just want to make sure this function doesn't crash on valid data
        try:
            analyze_correlations(self.mock_data)
        except Exception as e:
            self.fail(f"analyze_correlations raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()