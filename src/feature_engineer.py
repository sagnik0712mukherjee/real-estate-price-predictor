import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def engineer_features(df):
    """
    Performs feature engineering including one-hot encoding for categorical variables.
    """
    logger.info("Performing feature engineering...")
    
    # Identify categorical columns to encode
    categorical_cols = ['furnishing', 'type', 'locality']
    
    # Filter to only existing columns
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    if not cols_to_encode:
        logger.warning("No categorical columns found for encoding.")
        return df

    # Perform OneHotEncoding (pd.get_dummies)
    df_encoded = pd.get_dummies(df, columns=cols_to_encode)
    
    logger.info(f"Feature engineering complete. New shape: {df_encoded.shape}")
    return df_encoded
