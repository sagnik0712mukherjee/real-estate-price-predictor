import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Loads raw data from a CSV file.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        pd.pandas.set_option('display.max_columns', None)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
