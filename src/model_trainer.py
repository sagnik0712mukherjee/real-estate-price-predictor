import tensorflow as tf
from tensorflow import keras
import logging
from config.config import LAYERS, OPTIMIZER, LOSS, EPOCHS, BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(input_shape):
    """
    Builds the ANN model based on configuration.
    """
    logger.info("Building model...")
    model = keras.Sequential([
        keras.layers.Dense(LAYERS[0], activation='sigmoid', input_shape=(input_shape,))
    ])
    
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=['accuracy'])
    return model

def train_model(X_train, y_train):
    """
    Trains the model on provided data.
    """
    input_shape = X_train.shape[1]
    model = build_model(input_shape)
    
    logger.info(f"Starting training for {EPOCHS} epochs...")
    model.fit(
        X_train.to_numpy(dtype='float32'), 
        y_train.to_numpy(dtype='float32'), 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    return model
