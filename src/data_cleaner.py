import pandas as pd
import logging
from config.config import DROP_COLUMNS, OTHERS_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(df):
    """
    Cleans the dataframe by dropping irrelevant columns and handling missing values.
    """
    logger.info("Cleaning data...")
    
    # Drop irrelevant columns defined in config
    df_cleaned = df.drop(columns=DROP_COLUMNS, errors='ignore')
    
    # Drop "others" column if it exists (as found in notebook visualization section)
    df_cleaned = df_cleaned.drop(columns=OTHERS_COLUMN, errors='ignore')
    
    # Handle missing values - in the notebook, it seems drops or defaults were used
    # df2 = df.drop(...) and then specific filtering for outliers like df9
    # For now, we follow the notebook's logical progression
    
    # Remove rows with null targets if any
    df_cleaned = df_cleaned.dropna(subset=['price'])
    
    logger.info(f"Cleaning complete. Remaining columns: {list(df_cleaned.columns)}")
    return df_cleaned
