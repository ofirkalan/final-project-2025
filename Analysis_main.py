import pandas as pd  # Importing pandas library for data handling
import seaborn as sns  # Importing seaborn for statistical data visualization
import matplotlib.pyplot as plt  # Importing matplotlib for plotting graphs
from sklearn.model_selection import train_test_split  # Importing function to split data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Importing scaler to normalize the feature values
from sklearn.linear_model import LogisticRegression  # Importing the Logistic Regression machine learning model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Importing metrics to evaluate model performance

# --- Helper Functions (פונקציות עזר שאפשר לבדוק) ---

def load_data():
    """Tries to import data from Text_read.py"""
    try:
        from Text_read import df_cleaned
        print("\n--- Successfully imported 'df_cleaned' from Text_read.py ---")
        return df_cleaned
    except ImportError:
        print("Error: Could not import 'Text_read.py'. Make sure both files are in the same folder.")
        return None

def preprocess_data(df, target_column='Diagnosis'):
    """
    Splits the dataframe into Features (X) and Target (y),
    and performs the train-test split.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
    X = df.drop(target_column, axis=1)  # X contains all columns except 'Diagnosis'
    y = df[target_column]  # y contains only the 'Diagnosis' column
    
    # Splitting the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Scales the data and trains a Logistic Regression model.
    Returns the trained model and the scaler.
    """
    scaler = StandardScaler()  # Creating a scaler object
    X_train_scaled = scaler.fit_transform(X_train)  # Fitting the scaler to the training data
    
    model = LogisticRegression()  # Creating the model
    model.fit(X_train_scaled, y_train)  # Training the model
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Predicts results for the test set and calculates accuracy.
    """
    X_test_scaled = scaler.transform(X_test)  # Transforming test data
    y_pred = model.predict(X_test_scaled)  # Generating predictions
    
    accuracy = accuracy_score(y_test, y_pred)  # Calculating accuracy
    return accuracy, y_pred

def get_feature_importance(model, feature_names):
    """
    Returns a dataframe of features sorted by their importance.
    """
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    feature_importance['Abs_Importance'] = feature_importance['Importance'].abs()
    return feature_importance.sort_values(by='Abs_Importance', ascending=False)

# --- Main Execution Block ---
# This block runs only if you run this file directly (not when importing it for tests)
if __name__ == "__main__":
    df_cleaned = load_data()
    
    if df_cleaned is not None:
        # 1. EDA
        correlation_matrix = df_cleaned.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Alzheimer\'s Data')
        plt.show()

        # 2. Preprocessing
        X_train, X_test, y_train, y_test = preprocess_data(df_cleaned)

        # 3. Training
        model, scaler = train_model(X_train, y_train)

        # 4. Evaluation
        accuracy, y_pred = evaluate_model(model, scaler, X_test, y_test)
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 5. Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # 6. Feature Importance
        feature_importance = get_feature_importance(model, X_train.columns)
        print("\nTop 5 Most Important Factors for Predicting Alzheimer's:")
        print(feature_importance[['Feature', 'Importance']].head(5))