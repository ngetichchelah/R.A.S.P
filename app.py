import streamlit as st
import joblib
import pandas as pd
import numpy as np
from feature_engineering import add_severe_accident_features

# Load all components
@st.cache_resource
def load_models():
    try:
        model = joblib.load('enhanced_accident_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        severity_mapping = {0: 'Serious', 1: 'Fatal', 2: 'Slight'}  # Create instead of loading
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, le, severity_mapping, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Load models
model, scaler, le, severity_mapping, feature_names = load_models()

def predict_accident_severity(input_data):
    if model is None:
        return "Model not loaded", [], 0
    
    try:
        # 1. Add severe accident features
        input_data_enhanced = add_severe_accident_features(input_data)
        
        # 2. Ensure all expected columns are present
        for feature in feature_names:
            if feature not in input_data_enhanced.columns:
                input_data_enhanced[feature] = 0  # Add missing columns
        
        # 3. Reorder columns to match training
        input_data_enhanced = input_data_enhanced[feature_names]
        
        # 4. Scale numerical features
        if 'speed_limit' in input_data_enhanced.columns:
            input_data_enhanced[['speed_limit']] = scaler.transform(input_data_enhanced[['speed_limit']])
        
        # 5. Predict
        prediction_encoded = model.predict(input_data_enhanced)[0]
        prediction_proba = model.predict_proba(input_data_enhanced)[0]
        
        # 6. Convert to readable format
        prediction_readable = severity_mapping[prediction_encoded]
        
        return prediction_readable, prediction_proba, prediction_encoded
    
    except Exception as e:
        return f"Error in prediction: {e}", [], 0

# Streamlit Interface
st.title("🚗 Accident Severity Prediction")
st.write("Predict the severity of road accidents based on various factors")

# Show model status
if model is not None:
    st.sidebar.success(" Model loaded successfully!")
    st.sidebar.write(f"Model type: {type(model).__name__}")
    st.sidebar.write(f"Features: {len(feature_names) if feature_names else 'Unknown'}")
    st.sidebar.write(f"Severity types: {list(severity_mapping.values())}")
else:
    st.sidebar.error(" Model failed to load")

# Input form
st.header("Enter Accident Details")

col1, col2 = st.columns(2)

with col1:
    speed_limit = st.number_input("Speed Limit (km/h)", min_value=0, max_value=200, value=50)
    
with col2:
    # Add more inputs as needed based on your model features
    st.write("More input fields can be added here based on your model's requirements")

# Prediction button
if st.button("Predict Accident Severity", type="primary"):
    if model is not None:
        # Create input dataframe
        input_data = pd.DataFrame({
            'speed_limit': [speed_limit],
            # Add other required features here with default values
        })
        
        # Make prediction
        with st.spinner("Making prediction..."):
            prediction, probabilities, encoded = predict_accident_severity(input_data)
        
        # Display results
        st.header("Prediction Results")
        
        if isinstance(prediction, str) and "Error" not in prediction:
            # Success
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Severity", prediction)
            
            with col2:
                if len(probabilities) > 0:
                    confidence = max(probabilities) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show probability breakdown
            if len(probabilities) > 0:
                st.subheader("Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Severity': list(severity_mapping.values()),
                    'Probability': probabilities
                })
                st.bar_chart(prob_df.set_index('Severity'))
        
        else:
            # Error
            st.error(f"Prediction failed: {prediction}")
    
    else:
        st.error("Cannot make prediction - model not loaded properly")

# Test section
st.header("Model Information")
if st.button("Test Model Loading"):
    if model is not None:
        st.success(" All components loaded successfully!")
        st.write(f"Model type: {type(model)}")
        if feature_names:
            st.write(f"Number of features: {len(feature_names)}")
            with st.expander("View all features"):
                st.write(feature_names)
    else:
        st.error("Model loading failed")

# Footer
st.markdown("---")
st.markdown("*Accident Severity Prediction Model*")