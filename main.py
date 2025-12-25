import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import DATA_PATH, TEST_SIZE
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.feature_engineer import engineer_features
from src.model_trainer import train_model
from src.predictor import predict_price

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Main orchestration function for the real estate price prediction pipeline.
    """
    try:
        # 1. Load Data
        df = load_data(DATA_PATH)
        
        # 2. Clean Data
        df_cleaned = clean_data(df)
        
        # 3. Feature Engineering
        # Before engineering, split features and target if needed, 
        # or engineer and then split. The notebook does get_dummies then splits.
        # We need to drop 'locality' separately because the notebook encodes it.
        # But wait, my engineer_features encodes it if it exists.
        
        df_engineered = engineer_features(df_cleaned)
        
        # 4. Split Inputs and Target
        if 'price' not in df_engineered.columns:
            logger.error("Target column 'price' missing after feature engineering.")
            return

        X = df_engineered.drop(columns=['price'])
        y = df_engineered['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
        
        # 5. Train Model
        model = train_model(X_train, y_train)
        
        # 6. Verification / Prediction
        predictions = predict_price(model, X_test.head(5))
        logger.info(f"Sample predictions: {predictions.flatten()}")
        logger.info(f"Actual values: {y_test.head(5).values}")
        
        logger.info("Pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
