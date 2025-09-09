import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform text by removing punctuation, stemming, and removing stopwords."""
    ps = PorterStemmer()
    # Convert to Lower
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove Non Alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove Stopwords and Punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    """Preprocess the DataFrame by transforming text and encoding target column. Removing Duplicate and transforming
    the text column."""
    try:
            logger.debug("Starting DataFrame preprocessing")
            # Encode target column
            encoder = LabelEncoder()
            df[target_column] = encoder.fit_transform(df[target_column])
            logger.debug(f"Encoded target column '{target_column}'")

            # Remove Duplicates rows
            df.drop_duplicates(keep='first', inplace=True)
            logger.debug("Removed duplicate rows")

            # Apply text transformation to the specified text column
            df.loc[:, text_column] = df[text_column].apply(transform_text)
            logger.debug(f"Transformed text column '{text_column}'")
            return df
    except KeyError as e:
        logger.error(f"Column not found during DataFrame preprocessing: {e}")
        raise 
    except Exception as e:
        logger.error(f"Error during DataFrame preprocessing: {e}")
        raise
nltk.download('punkt_tab')

def main(text_column='text', target_column='target'):
    """Main function to load, preprocess, split, and save the data."""
    try:
        # Fetch the data from data/raw
        train_data=pd.read_csv('data/raw/train.csv')
        test_data=pd.read_csv('data/raw/test.csv')
        logger.debug("Data loaded successfully from data/raw")

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the Data inside data/processed
        data_path=os.path.join('data', 'interim')   
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug(f"Processed data saved successfully at {data_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No Data: {e}")

def main(text_column='text', target_column='target'):
    """Main function to load, preprocess, split, and save the data."""
    try:
        # Fetch the data from data/raw
        train_data=pd.read_csv('data/raw/train.csv')
        test_data=pd.read_csv('data/raw/test.csv')
        logger.debug("Data loaded successfully from data/raw")

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the Data inside data/processed
        data_path=os.path.join('data', 'interim')   
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug(f"Processed data saved successfully at {data_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No Data: {e}")
        raise    
    except Exception as e:
        logger.error(f"Failed to complete the Data Transformation Process: {e}")
        raise

if __name__ == "__main__":
    main()   