from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
import logging

logger = logging.getLogger(__name__)

def train_logistic_regression(X_train, y_train):
    """Trains Logistic Regression with balanced weights."""
    try:
        logger.info("Training Logistic Regression...")
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error training LR: {e}")
        raise

def train_random_forest(X_train, y_train):
    """Trains Random Forest with balanced weights."""
    try:
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,        # יותר עצים לדיוק טוב יותר
            max_depth=8,             # עומק מוגבל כדי למנוע שינון
            class_weight='balanced', # קריטי לזיהוי חולים
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error training RF: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_name="Model"):
    try:
        logger.info(f"--- Evaluating {model_name} ---")
        y_pred = model.predict(X_test)
        
        # המדד הכי חשוב לנו: Recall (כמה חולים תפסנו)
        recall = recall_score(y_test, y_pred, pos_label=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"{model_name} Recall (Sensitivity): {recall * 100:.2f}%")
        logger.info(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
        
        return recall
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        raise