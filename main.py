import logging
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.modeling import train_random_forest, evaluate_model # אנחנו נשארים עם RF כי הוא חזק יותר
from src.visualization import plot_confusion_matrix, plot_feature_importance, plot_lifestyle_cognition_correlation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. טעינת הנתונים
        df = load_and_clean_data('alzheimers_disease_data.csv')

        # --- חלק א: מענה לשאלה משנית (קורלציה לאורח חיים) ---
        print("\n--- Analyzing Research Question 2: Lifestyle vs Cognition ---")
        plot_lifestyle_cognition_correlation(df)
        
        # --- חלק ב: מענה לשאלה ראשית (חיזוי ללא דמוגרפיה) ---
        print("\n--- Preparing Data for Research Question 1: Prediction ---")
        # העיבוד מוחק עכשיו את הגיל, המגדר והתסמינים
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

        # אימון המודל (Random Forest)
        model = train_random_forest(X_train, y_train)
        
        # הערכה
        evaluate_model(model, X_test, y_test, "Random Forest (Lifestyle & Clinical Only)")

        # ויזואליזציה של המודל
        print("\n--- Displaying Prediction Results ---")
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred)
        plot_feature_importance(model, feature_names)

        logger.info("Pipeline finished.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()