import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout

# Create model directory if it doesn't exist
os.makedirs('model_train', exist_ok=True)

# Load the sample dataset
def load_data(file_path='Train_Data/Tomato_data.csv'):
    data = pd.read_csv('Train_Data/Tomato_data.csv')
    return data

# Preprocess data
def preprocess_data(data):
    # For this example, we'll assume the last column is the target variable
    # and all other columns are features
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Check if y contains string values
    if isinstance(y[0], str):
        # Convert string labels to numeric values
        # Assuming the labels are: "Stress", "Healthy", "Uncertain", "Recovery"
        label_mapping = {
            "Stress": 0,
            "Healthy": 1,
            "Uncertain": 2,
            "Recovery": 3
        }
        
        # Convert string labels to numeric using the mapping
        y_numeric = np.array([label_mapping.get(label, 0) for label in y])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)
    else:
        # Try to convert to numeric if possible
        try:
            y = pd.to_numeric(y, errors='coerce')
            # Convert to binary classification for simplicity
            # Assuming negative values indicate water stress and positive values indicate no stress
            y_binary = (y > 0).astype(int)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        except:
            # If conversion fails, raise a more helpful error
            raise TypeError("Target variable contains values that cannot be compared with numbers. Please check your data format.")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    with open('model_train/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train and save Decision Tree model
def train_decision_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Save the model
    with open('model_train/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    
    return dt_model

# Train and save Random Forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save the model
    with open('model_train/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    return rf_model

# Reshape data for CNN and LSTM models
def reshape_for_deep_learning(X_train, X_test):
    # Reshape for CNN and LSTM: [samples, time steps, features]
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train_reshaped, X_test_reshaped

# Train and save CNN model
def train_cnn(X_train_reshaped, y_train):
    # Define CNN model
    cnn_model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    cnn_history = cnn_model.fit(
        X_train_reshaped, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    cnn_model.save('model_train/cnn_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['accuracy'])
    plt.plot(cnn_history.history['val_accuracy'])
    plt.title('CNN Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['loss'])
    plt.plot(cnn_history.history['val_loss'])
    plt.title('CNN Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('model_train/cnn_training_history.png')
    
    return cnn_model

# Train and save LSTM model
def train_lstm(X_train_reshaped, y_train):
    # Define LSTM model
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    lstm_history = lstm_model.fit(
        X_train_reshaped, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    lstm_model.save('model_train/lstm_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['accuracy'])
    plt.plot(lstm_history.history['val_accuracy'])
    plt.title('LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history.history['loss'])
    plt.plot(lstm_history.history['val_loss'])
    plt.title('LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('model_train/lstm_training_history.png')
    
    return lstm_model

# Evaluate models
def evaluate_models(models, X_test, X_test_reshaped, y_test):
    results = {}
    
    # Evaluate Decision Tree
    dt_pred = models['decision_tree'].predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_report = classification_report(y_test, dt_pred)
    results['Decision Tree'] = {
        'accuracy': dt_accuracy,
        'report': dt_report
    }
    
    # Evaluate Random Forest
    rf_pred = models['random_forest'].predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred)
    results['Random Forest'] = {
        'accuracy': rf_accuracy,
        'report': rf_report
    }
    
    # Evaluate CNN
    cnn_pred = (models['cnn'].predict(X_test_reshaped) > 0.5).astype(int)
    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    cnn_report = classification_report(y_test, cnn_pred)
    results['CNN'] = {
        'accuracy': cnn_accuracy,
        'report': cnn_report
    }
    
    # Evaluate LSTM
    lstm_pred = (models['lstm'].predict(X_test_reshaped) > 0.5).astype(int)
    lstm_accuracy = accuracy_score(y_test, lstm_pred)
    lstm_report = classification_report(y_test, lstm_pred)
    results['LSTM'] = {
        'accuracy': lstm_accuracy,
        'report': lstm_report
    }
    
    # Save evaluation results
    with open('model_train/evaluation_results.txt', 'w') as f:
        for model_name, result in results.items():
            f.write(f"{model_name} Accuracy: {result['accuracy']}\n")
            f.write(f"{model_name} Classification Report:\n")
            f.write(f"{result['report']}\n\n")
    
    # Create comparison bar chart
    accuracies = [results[model]['accuracy'] for model in results]
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), accuracies, color=['blue', 'green', 'red', 'purple'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig('model_train/model_comparison.png')
    
    return results

# Main function to train all models
def train_all_models():
    print("Loading and preprocessing data...")
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    print("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train)
    
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Reshaping data for deep learning models...")
    X_train_reshaped, X_test_reshaped = reshape_for_deep_learning(X_train, X_test)
    
    print("Training CNN model...")
    cnn_model = train_cnn(X_train_reshaped, y_train)
    
    print("Training LSTM model...")
    lstm_model = train_lstm(X_train_reshaped, y_train)
    
    print("Evaluating all models...")
    models = {
        'decision_tree': dt_model,
        'random_forest': rf_model,
        'cnn': cnn_model,
        'lstm': lstm_model
    }
    
    results = evaluate_models(models, X_test, X_test_reshaped, y_test)
    
    print("Training and evaluation completed. Models saved in 'model_train' directory.")
    
    return models, results

if __name__ == "__main__":
    train_all_models()



