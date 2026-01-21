# Alzheimer's Disease Prediction & Analysis System

## 1. Project Description & Objectives

**Description:**
This project aims to analyze and predict Alzheimer's disease diagnosis based on patient medical records and demographic data. The system utilizes a modular data pipeline to ingest raw data, perform statistical analysis, and train machine learning models to identify key risk factors and classify patients effectively.

**Main Objectives:**
1.  To identify significant correlations between demographic/medical features (such as Age and Ethnicity) and Alzheimer's diagnosis.
2.  To compare the performance of linear models (Logistic Regression) versus non-linear ensemble models (Random Forest) in a medical context.

**Hypothesis:**
We hypothesize that Age is the primary risk factor, but non-linear relationships exist between demographic subgroups and diagnosis rates. Therefore, we expect the Random Forest model to outperform Logistic Regression by better capturing these complex interactions and handling class imbalances in the medical data.

**Key Assumptions:**
* **Data Integrity:** We assume that the provided dataset is a representative sample of the population.
* **Missing Values:** We assume that rows containing missing values (NaN) are missing at random and can be safely removed without introducing significant bias to the model.
* **Independence:** We assume that patient records are independent of one another.

## 2. Project Structure

The project is organized into a modular structure to ensure readability and separation of concerns:

```text
final-project-2025/
│
├── src/                            # Source code directory
│   ├── alzheimers_disease_data.csv # The dataset file
│   ├── Main_code.py                # Entry point: Orchestrates the entire pipeline
│   ├── Main_plots.py               # Visualization functions
│   ├── Statistic_analysis.py       # Statistical tests (Correlations, P-Values)
│   ├── Statistical_model.py        # Feature Engineering, Preprocessing, and Model Training
│   ├── Test_analysis.py            # Unit tests for the project
│   └── Text_read.py                # Data ingestion and initial cleaning
│
├── logs/                           # Automated runtime logs (created upon execution)
├── requirements.txt                # List of dependencies
└── README.md                       # Project documentation
```

## 3. Key Stages & Workflow
The project follows a structured data analysis pipeline (Data Import → Processing → Analysis → Modeling → Visualization):

Data Import:

Loading raw data (alzheimers_disease_data.csv) using Text_read.py.

Feature Engineering & Processing:

Age Binning: Converting continuous Age into categorical groups (60-70, 70-80, etc.) to handle non-linear risk stratification.

One-Hot Encoding: Converting categorical variables (e.g., Ethnicity) into binary vectors.

Cleaning: Dropping irrelevant columns and removing NaN values.

Splitting: Dividing data into Train/Test sets (80/20 split).

Statistical Analysis:

Performing P-Value tests (in Statistic_analysis.py) to validate the significance of specific features before modeling.

Modeling:

Logistic Regression: Trained with StandardScaler to serve as a linear baseline.

Random Forest: Trained with class_weight='balanced' to capture complex patterns.

Visualization:

Generating Confusion Matrices to evaluate accuracy.

Plotting Feature Importance to interpret the Random Forest's decision-making logic.

## 4. Key Configurations & Parameters
The following parameters are critical for the model's performance and reproducibility:

random_state = 42: Ensures that data splitting and model initialization are reproducible across different runs.

test_size = 0.2: We reserve 20% of the data for testing to validate the model on unseen data.

class_weight = 'balanced': Used in Random Forest. This adjusts weights inversely proportional to class frequencies, helping the model pay more attention to the minority class (Positive Diagnosis).

n_estimators = 100: The number of trees in the Random Forest.

Age Bins [60, 70, 80, 90, 100]: Defined to align with standard geriatric risk groups.

## 5. Dataset Description
Source: Included in the src folder (alzheimers_disease_data.csv).

Description: The dataset consists of anonymized patient health records used for classification tasks.

Target Variable: Diagnosis (0 = Negative, 1 = Positive).

Key Features:

Age: Continuous variable (processed into bins).

Ethnicity: Categorical demographic data.

Medical History: Various clinical indicators.

## 6. How to Run (Instructions)
To run this project on your local machine, follow these steps:

Install Dependencies: Make sure you are in the project root folder and run:

Bash

pip install -r requirements.txt
Run the Analysis: Execute the main script from the src folder:

## Bash

python src/Main_code.py

Run Tests: To verify the statistical components:

## Bash

python src/Test_analysis.py
## 7. References
Scikit-Learn Documentation: https://scikit-learn.org/stable/

Pandas Documentation: https://pandas.pydata.org/docs/

Alzheimer's Association: Kaggle Alzheimer's Dataset
