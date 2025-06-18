import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

def load_and_validate_data(file_path):
    """
    Load data from CSV or Excel file and perform basic validation.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Validated dataframe or None if validation fails
    """
    try:
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel file."
        
        # Basic validation
        if data.empty:
            return None, "The uploaded file contains no data."
        
        # Check for missing values
        missing_percentage = data.isnull().mean() * 100
        if missing_percentage.max() > 20:
            return data, f"Warning: Some columns have more than 20% missing values. Max missing: {missing_percentage.max():.1f}%"
        
        return data, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def preprocess_data(data, scaler_type='standard', save_scaler=True):
    """
    Preprocess the data by handling missing values and scaling.
    
    Args:
        data (pd.DataFrame): Input dataframe
        scaler_type (str): Type of scaling ('standard' or 'minmax')
        save_scaler (bool): Whether to save the scaler for later use
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
        object: Fitted scaler object
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Select numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Initialize the appropriate scaler
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Scaler type must be 'standard' or 'minmax'")
    
    # Fit and transform the data
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save the scaler if requested
    if save_scaler:
        os.makedirs('model_train', exist_ok=True)
        joblib.dump(scaler, 'model_train/scaler.pkl')
    
    return df, scaler

def extract_features(data, window_size=10, step_size=1):
    """
    Extract time-series features using a sliding window approach.
    
    Args:
        data (pd.DataFrame): Input dataframe
        window_size (int): Size of the sliding window
        step_size (int): Step size for the sliding window
        
    Returns:
        np.ndarray: Feature matrix X
        np.ndarray: Target vector y (if available)
    """
    # Assume all columns except the last one are features
    if data.shape[1] > 1:
        X_data = data.iloc[:, :-1].values
        y_data = data.iloc[:, -1].values if data.shape[1] > 1 else None
    else:
        X_data = data.values
        y_data = None
    
    # Create sliding windows
    X_windows = []
    y_windows = []
    
    for i in range(0, len(X_data) - window_size + 1, step_size):
        X_windows.append(X_data[i:i+window_size])
        if y_data is not None:
            y_windows.append(y_data[i+window_size-1])
    
    X = np.array(X_windows)
    y = np.array(y_windows) if y_data is not None else None
    
    return X, y

def prepare_data_for_model(data, model_type, test_size=0.2, random_state=42):
    """
    Prepare data for specific model types.
    
    Args:
        data (pd.DataFrame): Preprocessed dataframe
        model_type (str): Type of model ('decision_tree', 'random_forest', 'lstm', 'cnn')
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Training and testing data in the format required by the model
    """
    from sklearn.model_selection import train_test_split
    
    # Extract features and target
    if data.shape[1] > 1:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    else:
        X = data.values
        y = None
    
    # Split data if target is available
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        # If no target, just split X for unsupervised learning
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        y_train, y_test = None, None
    
    # Reshape data for deep learning models
    if model_type.lower() in ['lstm', 'cnn']:
        # Reshape to [samples, time steps, features]
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test

def generate_sample_data(n_samples=1000, n_features=5, random_state=42):
    """
    Generate sample data for demonstration purposes.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated dataframe
    """
    np.random.seed(random_state)
    
    # Generate feature data
    data = np.random.randn(n_samples, n_features)
    
    # Create time index
    date_rng = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f'sensor{i+1}' for i in range(n_features)])
    df['timestamp'] = date_rng
    
    # Add a target column (0: Healthy, 1: Stress, 2: Uncertain, 3: Recovery)
    stress_pattern = np.zeros(n_samples)
    
    # Create patterns of stress
    for i in range(n_samples):
        if i % 200 < 50:  # Stress periods
            stress_pattern[i] = 1
        elif i % 200 < 70:  # Recovery periods
            stress_pattern[i] = 3
        elif i % 200 < 80:  # Uncertain periods
            stress_pattern[i] = 2
        else:  # Healthy periods
            stress_pattern[i] = 0
    
    df['status'] = stress_pattern
    
    # Add some noise to make it more realistic
    for col in df.columns[:-2]:  # Exclude timestamp and status
        df[col] = df[col] + 0.5 * np.sin(np.linspace(0, 10*np.pi, n_samples))
    
    # Add some missing values
    mask = np.random.random(df.shape) < 0.05
    df.mask(mask, inplace=True)
    
    return df