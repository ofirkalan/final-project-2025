# final-project-2025
Final project of python course
# Alzheimer's Disease Prediction & Analysis System

## 1. Project Description & Objectives

**Description:**
This project aims to analyze and predict Alzheimer's disease diagnosis based on patient medical records and demographic data. The system utilizes a modular data pipeline to ingest raw data, perform statistical analysis, and train machine learning models to identify key risk factors and classify patients effectively.

**Main Objectives:**
1.  To identify significant correlations between demographic/medical features (such as Age and Ethnicity) and Alzheimer's diagnosis.
2.  To compare the performance of linear models (Logistic Regression) versus non-linear ensemble models (Random Forest) in a medical context.

**Hypothesis:**
We hypothesize that Age is the primary risk factor, but non-linear relationships exist between demographic subgroups and diagnosis rates. Therefore, we expect the Random Forest model to outperform Logistic Regression by better capturing these complex interactions and handling class imbalances in the medical data.

## 2. Dataset Description

**Source:** Public Medical Health Records (e.g., Kaggle / OASIS)

**Description:**
The dataset consists of anonymized patient health records used for classification tasks.
* **Target Variable:** `Diagnosis` (0 = Negative, 1 = Positive).
* **Key Features:**
    * `Age`: Continuous variable (processed into bins).
    * `Ethnicity`: Categorical demographic data.
    * `Medical History`: Various clinical indicators.
* **Preprocessing:** Rows containing missing values (NaN) are removed to ensure model stability.

## 3. Project Structure

The project is organized into a modular structure to ensure readability and separation of concerns:

```text
├── src/
│   ├── main.py                 # Entry point: Orchestrates the entire pipeline
│   ├── Text_read.py            # Data ingestion and initial cleaning
│   ├── Statistic_analysis.py   # Statistical tests (Correlations, P-Values)
│   ├── Statistical_model.py    # Feature Engineering and Model Training logic
│   └── Main_plots.py           # Visualization functions
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
