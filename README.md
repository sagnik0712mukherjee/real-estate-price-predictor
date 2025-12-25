# Mumbai Real Estate Price Predictor

A production-grade implementation of an Artificial Neural Network (ANN) to predict house prices in various Mumbai localities. This project has been refactored from a monolithic Jupyter Notebook into a modular, scalable Python package.

## Project Features

- **Modular Architecture**: Separate modules for data loading, cleaning, feature engineering, and model training.
- **Deep Learning Pipeline**: Built using TensorFlow/Keras for building and training predictive models.
- **Configurable**: Easy-to-adjust hyperparameters and data schema via `config/config.py`.
- **Production Ready**: Includes comprehensive logging, error handling, and a clear execution entry point.

## Repository Structure

```text
real-estate-price-predictor/
├── src/
│   ├── data_loader.py        # Generic CSV data loading
│   ├── data_cleaner.py      # Automated cleaning and column reduction
│   ├── feature_engineer.py   # One-hot encoding for categorical variables
│   ├── model_trainer.py      # ANN architecture and training logic
│   └── predictor.py           # Inference wrapper
├── config/
│   └── config.py              # Centralized hyperparameters and paths
├── main.py                    # Pipeline orchestration entry point
├── requirements.txt           # Python dependencies
├── .gitignore                 # Standard exclusions
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sagnik0712mukherjee/real-estate-price-predictor.git
   cd real-estate-price-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the full training pipeline and see sample predictions:
```bash
python main.py
```

## Model Details

The current implementation uses a Keras Sequential model with:
- **Architecture**: 1 Hidden layer with 100 units (Sigmoid activation).
- **Optimizer**: RMSprop.
- **Loss Function**: Categorical Crossentropy (Replicated from original research).
- **Target Variable**: House Price (Mumbai Localities).

## Author

**Sagnik Mukherjee**  
[GitHub Profile](https://github.com/sagnik0712mukherjee)