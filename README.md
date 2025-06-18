# Tomato Water Stress Prediction System

An AI-based application for classifying and forecasting water stress in tomato plants using bioristor sensor data.

## Project Overview

This system uses machine learning and deep learning models to predict water stress in tomato plants, enabling data-driven irrigation decisions to improve crop yield and water efficiency.

## Key Features

- **Classification Models**: Decision Trees, Random Forest
- **Prediction Models**: LSTM and CNN neural networks
- **High Accuracy**: Models achieve up to 97% accuracy in predicting water stress
- **User-friendly Interface**: Streamlit web application for easy data upload and analysis
- **Smart Irrigation Support**: Data-driven irrigation recommendations

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Visualization**: Matplotlib, Plotly

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/kusetti143/Predicting-Water-Stress
   cd Predicting-Water-Stress
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or
   if system is not able to install upgrade pip using 
   ```
   python -m pip install --upgrade 
   ```

## Usage

1. **Train the models** (if not already trained):
   ```
   python train_models.py
   ```

2. **Run the application**:
   ```
   streamlit run app.py
   ```

3. **Navigate through the application**:
   - Upload bioristor sensor data
   - Select a model for prediction
   - View results and recommendations

## Project Structure

- `app.py`: Main Streamlit application
- `train_models.py`: Script to train and save ML/DL models
- `utils/preprocessing.py`: Data preprocessing utilities
- `model_train/`: Directory containing trained models and evaluation results
- `Datasets/`: Sample datasets for testing

## Models

- **Decision Tree**: Classification model for water stress detection
- **Random Forest**: Ensemble model for improved classification accuracy
- **CNN**: Convolutional Neural Network for pattern recognition in sensor data
- **LSTM**: Long Short-Term Memory network for time-series prediction

## Results

The models have been evaluated on test data with the following metrics:
- Accuracy: 97%
- Precision: 95%
- Recall: 94%
- F1 Score: 94.5%

## Future Work

- Integration with IoT devices for real-time monitoring
- Mobile application development
- Expansion to other crop types
- Improved visualization of prediction results


## Contributors

- K.Balaji


## Acknowledgments

- Used Kaggle Datasets for this project

