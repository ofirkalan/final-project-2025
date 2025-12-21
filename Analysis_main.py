import pandas as pd  # Importing pandas library for data handling
import seaborn as sns  # Importing seaborn for statistical data visualization
import matplotlib.pyplot as plt  # Importing matplotlib for plotting graphs
from sklearn.model_selection import train_test_split  # Importing function to split data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Importing scaler to normalize the feature values
from sklearn.linear_model import LogisticRegression  # Importing the Logistic Regression machine learning model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Importing metrics to evaluate model performance

# --- 1. Importing Cleaned Data from Text_read.py ---

# This is the key line: it runs Text_read.py and imports the 'df_cleaned' variable from it
try:
    from Text_read import df_cleaned
    print("\n--- Successfully imported 'df_cleaned' from Text_read.py ---")
except ImportError:
    print("Error: Could not import 'Text_read.py'. Make sure both files are in the same folder.")
    exit() # Stop the script if import fails

# --- 2. Exploratory Data Analysis (EDA) ---

# We want to see how variables correlate with the Diagnosis
# Calculating the correlation matrix for all columns
correlation_matrix = df_cleaned.corr()  # Computes pairwise correlation of columns

# Plotting the correlation heatmap
plt.figure(figsize=(12, 10))  # Sets the size of the figure (width, height)
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)  # Creates a heatmap with a red-blue color scheme
plt.title('Correlation Matrix of Alzheimer\'s Data')  # Sets the title of the plot
plt.show()  # Displays the heatmap to the screen

# --- 3. Data Preprocessing ---

# Separating the Target (what we want to predict) from the Features (data used for prediction)
target_column = 'Diagnosis'  # Defining the name of the target column
X = df_cleaned.drop(target_column, axis=1)  # X contains all columns except 'Diagnosis'
y = df_cleaned[target_column]  # y contains only the 'Diagnosis' column

# Splitting the data: 80% for training the model, 20% for testing it later
# random_state=42 ensures the split is reproducible (same result every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data (Scaling)
# This ensures that variables with large ranges (like Cholesterol) don't dominate variables with small ranges (like Age)
scaler = StandardScaler()  # Creating a scaler object
X_train_scaled = scaler.fit_transform(X_train)  # Fitting the scaler to the training data and transforming it
X_test_scaled = scaler.transform(X_test)  # Transforming the test data using the scale learned from training data

# --- 4. Model Training ---

# We define the Logistic Regression model
model = LogisticRegression()  # Creating an instance of the Logistic Regression model

# We train the model using the scaled training data and the known answers (y_train)
model.fit(X_train_scaled, y_train)  # The model learns the relationship between X and y

# --- 5. Model Evaluation ---

# Now we ask the model to predict the diagnosis for the test set (which it hasn't seen yet)
y_pred = model.predict(X_test_scaled)  # Generating predictions for the test data

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)  # Comparing predicted values to actual values
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")  # Printing the accuracy as a percentage

# Generate a detailed classification report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Printing detailed performance metrics

# --- 6. Visualizing Results (Confusion Matrix) ---

# Creating a confusion matrix to see True Positives, False Positives, etc.
conf_matrix = confusion_matrix(y_test, y_pred)  # Calculating the confusion matrix

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))  # Setting the figure size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # Creating a heatmap with actual numbers ('d')
plt.xlabel('Predicted Label')  # Label for x-axis
plt.ylabel('True Label')  # Label for y-axis
plt.title('Confusion Matrix')  # Title of the plot
plt.show()  # Displaying the plot

# --- 7. Feature Importance ---

# Identifying which features contributed most to the prediction
# Getting the coefficients from the trained model
coefficients = model.coef_[0]

# Creating a DataFrame to display features and their importance scores
feature_importance = pd.DataFrame({
    'Feature': X.columns,  # Column names
    'Importance': coefficients  # Model weights (coefficients)
})

# Sorting the features by their absolute importance (magnitude of impact)
feature_importance['Abs_Importance'] = feature_importance['Importance'].abs()  # Calculating absolute value
feature_importance = feature_importance.sort_values(by='Abs_Importance', ascending=False)  # Sorting descending

# Printing the top 5 most important factors
print("\nTop 5 Most Important Factors for Predicting Alzheimer's:")
print(feature_importance[['Feature', 'Importance']].head(5))  # Displaying the top 5 rows