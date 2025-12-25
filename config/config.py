import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "prop_data_clean.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "mumbai_real_estate_model.h5")

# Dataset columns
DROP_COLUMNS = [
    'id', 'desc', 'city', 'trans', 'dev_name', 'id_string', 
    'latitude', 'longitude', 'post_date', 'user_type', 
    'poster_name', 'project', 'title', 'url'
]

# Additional drops found in notebook analysis
OTHERS_COLUMN = ['others']

# Model Hyperparameters
LAYERS = [100]
OPTIMIZER = 'rmsprop'
LOSS = 'categorical_crossentropy'
EPOCHS = 5
BATCH_SIZE = 32
TEST_SIZE = 0.2
