import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import json
import numpy as np

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load data from a CSV file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Data loaded successfully from {file_path}")
        return model    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill NaN values with empty strings
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and save the classification report.
    : param clf: Trained RandomForestClassifier model
    : param x_test: Test features
    : param y_test: Test labels
    : param report_path: Path to save the classification report
    """
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': auc
        }
        logger.info(f"Model evaluation metrics Calculated: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the classification report to a JSON file.
    : param metrics: Dictionary containing evaluation metrics
    : param file_path: Path to save the classification report
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Classification report saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving classification report to {file_path}: {e}")
        raise 

def main():
    """Main function to load, evaluate, and save the model metrics."""
    try:
        # Load the test data
        test_data = load_data('data/processed/test_tfidf.csv')
        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Load the trained model
        clf = load_model('models/random_forest_model.pkl')

        # Evaluate the model
        metrics = evaluate_model(clf, x_test, y_test)

        # Save the evaluation metrics
        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise       

if __name__ == "__main__":
    main()    