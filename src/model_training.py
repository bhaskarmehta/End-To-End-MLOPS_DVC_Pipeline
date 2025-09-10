import os
import logging
import numpy as np
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
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

def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """Train a RandomForestClassifier model.
    : param x_train: Training features
    : param y_train: Training labels
    : param params: Dictionary to Hyperparameters for the RandomForestClassifier
    : return: Trained RandomForestClassifier model
    """
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in x_train and y_train must be equal.")
        logger.debug("Initializing RandomForestClassifier with parameters: {}".format(params))
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug("Starting model training with {} samples.".format(x_train.shape[0]))
        clf.fit(x_train, y_train)
        logger.info("Model training completed successfully.")
        return clf
    except ValueError as ve:
        logger.error(f"ValueError during model training: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model: RandomForestClassifier, file_path: str) -> None: 
    """Save the trained model to a file.
    : param model: Trained RandomForestClassifier model
    : param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully at {file_path}")
    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        raise    
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise   

def main():
    try:
        # Load the transformed data
        train_data = load_data('data/processed/train_tfidf.csv')

        # Separate features and labels
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        # Define model hyperparameters
        # params = {
        #     'n_estimators': 25,
        #     'random_state': 2
        # }

        params = load_params(params_path='params.yaml')['model_training']
        # params['n_estimators'] = params['model_training']['n_estimators']
        # params['random_state'] = params['model_training']['random_state']

        clf = train_model(x_train, y_train, params)
        # Save the trained model
        model_save_path = 'models/random_forest_model.pkl'
        save_model(clf, model_save_path)   
    except Exception as e:
        logger.error(f"Failed to complete the model training process: {e}")
        raise

if __name__ == "__main__":
    main()        