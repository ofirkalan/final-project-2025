import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_test, y_pred):
    """Plots the confusion matrix heatmap."""
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Predicting Diagnosis)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning(f"Could not plot confusion matrix: {e}")

def plot_feature_importance(model, feature_names):
    """Plots the top 12 most important features."""
    try:
        importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Abs_Importance': abs(importance)
        })
        
        feature_importance = feature_importance.sort_values(by='Abs_Importance', ascending=False).head(12)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Top Factors Predicting Alzheimer\'s (Excluding Demographics)')
        plt.xlabel('Influence on Prediction (Left=Protective, Right=Risk)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")

def plot_lifestyle_cognition_correlation(df):
    """
    Answers Research Question 2:
    Visualizes correlation between Lifestyle Factors and Cognitive Score (MMSE).
    Explicitly removes the MMSE row.
    """
    try:
        # בחירת העמודות
        cols_of_interest = [
            'DietQuality', 'SleepQuality', 'PhysicalActivity', 
            'BMI', 'AlcoholConsumption', 'Smoking',
            'MMSE'
        ]
        
        available_cols = [c for c in cols_of_interest if c in df.columns]
        
        # חישוב קורלציה
        corr_matrix = df[available_cols].corr()
        
        # --- השינוי המובטח ---
        # אנחנו מסננים החוצה את שורת ה-MMSE באופן מפורש
        if 'MMSE' in corr_matrix.index:
            print("DEBUG: Removing MMSE row from plot...") # הודעה שתוודא שהקוד החדש רץ
            data_to_plot = corr_matrix[['MMSE']].drop(index='MMSE')
        else:
            data_to_plot = corr_matrix[['MMSE']]
            
        # מיון התוצאות
        data_to_plot = data_to_plot.sort_values(by='MMSE', ascending=False)
        
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(data_to_plot,
                              annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation of Lifestyle Factors with Cognitive Score (MMSE)')
        plt.tight_layout()
        plt.show()
        logger.info("Plotted Lifestyle vs MMSE correlation.")
        
    except Exception as e:
        logger.warning(f"Could not plot lifestyle correlation: {e}")