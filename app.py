import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os
import json

# Configure Streamlit for production
st.set_page_config(
    page_title="Tomato Water Stress Prediction",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Import TensorFlow only when needed to reduce cold start time
@st.cache_resource
def load_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        st.error("TensorFlow not available. Deep learning models will not work.")
        return None

# Cache model loading
@st.cache_resource
def load_model(model_type):
    try:
        if model_type in ["Decision Tree", "Random Forest"]:
            model_path = f'model_train/{model_type.lower().replace(" ", "_")}_model.pkl'
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif model_type in ["CNN", "LSTM"]:
            tf = load_tensorflow()
            if tf is None:
                return None
            model_path = f'model_train/{model_type.lower()}_model.h5'
            return tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        st.error(f"{model_type} model not found.")
        return None
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('model_train/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("Scaler not found. Using unscaled data.")
        return None

# Main application
def main():
    # Sidebar for navigation
    st.sidebar.title("🍅 Navigation")
    
    page = st.sidebar.radio("Go to", ["Home", "Data Upload", "Prediction", "Results", "About"])
    
    if page == "Home":
        show_home()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "Prediction":
        show_prediction()
    elif page == "Results":
        show_results()
    elif page == "About":
        show_about()

# Home page
def show_home():
    st.title("🍅 AI-based Tomato Water Stress Prediction")
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff6b6b, #4ecdc4); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center;">Welcome to the Tomato Water Stress Prediction System</h2>
        <p style="color: white; text-align: center; font-size: 1.2rem;">
            AI-powered solution for smart irrigation and crop management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Classification Models**: Decision Trees, Random Forest
        - **Prediction Models**: LSTM and CNN
        - **High Accuracy**: Up to 97% accuracy in predictions
        - **Smart Irrigation**: Data-driven irrigation decisions
        - **Real-time Analysis**: Process bioristor sensor data
        """)
    
    with col2:
        st.subheader("🚀 How to Use")
        st.markdown("""
        1. **Upload Data**: Upload your bioristor sensor data
        2. **Select Model**: Choose from 4 AI models
        3. **Get Predictions**: Analyze water stress levels
        4. **View Results**: Get irrigation recommendations
        5. **Take Action**: Implement smart irrigation
        """)

def show_data_upload():
    st.title("📊 Data Upload")
    st.markdown("Upload your bioristor sensor data for water stress analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload CSV file containing bioristor sensor data"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
            st.success("✅ Data uploaded successfully!")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Size", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Display data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Basic statistics
            if st.checkbox("Show Statistics"):
                st.subheader("📈 Data Statistics")
                st.dataframe(data.describe(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_prediction():
    st.title("🔮 Water Stress Prediction")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data first!")
        st.info("Go to the 'Data Upload' page to upload your sensor data.")
        return
    
    data = st.session_state['data']
    
    # Model selection
    st.subheader("🤖 Select AI Model")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Choose Model", 
            ["Decision Tree", "Random Forest", "LSTM", "CNN"],
            help="Select the AI model for prediction"
        )
    
    with col2:
        st.info(f"Selected: **{model_type}**")
    
    # Model descriptions
    model_descriptions = {
        "Decision Tree": "Fast, interpretable classification model",
        "Random Forest": "Ensemble model with high accuracy",
        "LSTM": "Deep learning model for time-series patterns",
        "CNN": "Convolutional network for pattern recognition"
    }
    
    st.markdown(f"*{model_descriptions[model_type]}*")
    
    # Run prediction
    if st.button("🚀 Run Prediction", type="primary"):
        run_prediction(data, model_type)

def run_prediction(data, model_type):
    with st.spinner(f"Running {model_type} prediction..."):
        try:
            # Define class labels
            class_labels = ["Stress", "Healthy", "Uncertain", "Recovery"]
            
            # Prepare features
            if data.shape[1] > 1:
                X = data.iloc[:, :-1].values
            else:
                X = data.values
            
            # Load and apply scaler
            scaler = load_scaler()
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Load model and make predictions
            model = load_model(model_type)
            if model is None:
                return
            
            if model_type in ["Decision Tree", "Random Forest"]:
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
            else:  # CNN or LSTM
                X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                probabilities = model.predict(X_reshaped)
                predictions = np.argmax(probabilities, axis=1)
            
            # Store results
            st.session_state.update({
                'predictions': predictions,
                'probabilities': probabilities,
                'model_used': model_type,
                'class_labels': class_labels
            })
            
            # Calculate overall prediction
            prediction_counts = np.bincount(predictions.astype(int), minlength=len(class_labels))
            majority_class = np.argmax(prediction_counts)
            prediction_result = class_labels[majority_class]
            confidence = prediction_counts[majority_class] / len(predictions)
            
            st.session_state.update({
                'prediction': prediction_result,
                'confidence': confidence
            })
            
            # Display results
            display_prediction_results(prediction_result, confidence, model_type, 
                                     class_labels, prediction_counts, predictions, probabilities)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

def display_prediction_results(prediction_result, confidence, model_type, 
                             class_labels, prediction_counts, predictions, probabilities):
    st.success("✅ Prediction completed!")
    
    # Main results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", prediction_result)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        st.metric("Model", model_type)
    
    # Visualization
    st.subheader("📊 Prediction Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_labels, prediction_counts / len(predictions))
    ax.set_ylabel('Proportion')
    ax.set_title('Water Stress Classification Results')
    ax.set_ylim(0, 1)
    
    # Color bars based on stress level
    colors = ['#ff4444', '#44ff44', '#ffaa44', '#4444ff']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    st.pyplot(fig)

def show_results():
    st.title("📈 Analysis Results")
    
    if 'prediction' not in st.session_state:
        st.warning("⚠️ No prediction results available.")
        st.info("Please run a prediction first!")
        return
    
    prediction = st.session_state['prediction']
    confidence = st.session_state['confidence']
    model_used = st.session_state['model_used']
    
    # Results summary
    st.subheader("🎯 Prediction Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction == "Stress":
            st.error(f"🚨 **Status**: {prediction}")
            st.error(f"🔥 **Water Stress Detected**")
        else:
            st.success(f"✅ **Status**: {prediction}")
            st.success(f"💧 **No Water Stress**")
    
    with col2:
        st.info(f"**Confidence**: {confidence:.1%}")
        st.info(f"**Model**: {model_used}")
    
    # Recommendations
    st.subheader("💡 Irrigation Recommendations")
    
    if prediction == "Stress":
        st.error("🚨 **IMMEDIATE ACTION REQUIRED**")
        st.markdown("""
        **Recommended Actions**:
        - 🚰 Increase irrigation frequency immediately
        - 📊 Monitor soil moisture levels closely
        - 🔍 Check irrigation system for proper distribution
        - 📱 Set up alerts for continuous monitoring
        """)
    else:
        st.success("✅ **NORMAL CONDITIONS**")
        st.markdown("""
        **Recommended Actions**:
        - 📅 Maintain current irrigation schedule
        - 👀 Continue regular monitoring
        - 💚 Consider water conservation if conditions remain favorable
        - 📈 Track trends for predictive maintenance
        """)

def show_about():
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🍅 AI-based Tomato Water Stress Prediction
    
    ### 🎯 Project Overview
    This advanced AI system classifies and forecasts water stress in tomato plants using 
    real-time bioristor sensor data, enabling smart irrigation decisions for improved 
    crop yield and water efficiency.
    
    ### 🛠️ Technologies Used
    - **Sensors**: Bioristor sensors for real-time plant monitoring
    - **ML Models**: Decision Trees, Random Forest
    - **DL Models**: LSTM and CNN neural networks
    - **Interface**: Streamlit web application
    - **Deployment**: Vercel cloud platform
    
    ### 🏆 Key Achievements
    - **97% Accuracy** in water stress prediction
    - **Real-time Processing** of sensor data
    - **Multi-model Approach** for robust predictions
    - **User-friendly Interface** for easy operation
    
    ### 🌱 Benefits
    - 💧 **Water Conservation**: Optimize irrigation schedules
    - 📈 **Increased Yield**: Prevent stress-related crop loss
    - 🌍 **Sustainability**: Support eco-friendly farming
    - 💰 **Cost Savings**: Reduce water and energy costs
    
    ### 👨‍💻 Developer
    **K.Balaji** - AI/ML Engineer
    
    ### 🙏 Acknowledgments
    - Kaggle community for datasets
    - Open source ML/DL libraries
    - Agricultural research community
    """)

if __name__ == "__main__":
    main()
