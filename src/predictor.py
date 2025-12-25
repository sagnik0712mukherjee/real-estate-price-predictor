import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_price(model, input_data):
    """
    Makes a price prediction using the trained model.
    """
    logger.info("Making prediction...")
    # Ensure input_data is in the right format (numpy array)
    if hasattr(input_data, 'to_numpy'):
        input_data = input_data.to_numpy(dtype='float32')
    
    prediction = model.predict(input_data)
    return prediction
