import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


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

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF vectorization to the text column of the train and test datasets."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        x_train_bow = vectorizer.fit_transform(X_train)
        x_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train 

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test  
        logger.debug("tfidf of word applied to training data") 
        return train_df,test_df    
    except Exception as e:
        logger.error(f"Error applying tfidf of word applied to training data: {e}")
        raise

def save_date(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    """Main function to load, transform, and save the data."""
    try:
        max_features = 50
        # Load the preprocessed data
        train_data = load_data('data/interim/train_processed.csv')
        test_data = load_data('data/interim/test_processed.csv')


        # Apply TF-IDF vectorization
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Save the transformed data
        save_date(train_df, 'data/processed/train_tfidf.csv')
        save_date(test_df, 'data/processed/test_tfidf.csv')
    except Exception as e:
        logger.error(f"Failed to complete the Feature Engineering: {e}")
        raise    

if __name__ == "__main__":
    main()
