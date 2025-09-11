import streamlit as st
import joblib
import pandas as pd
import numpy as np
from feature_engineering import add_severe_accident_features  # Your saved function

# Load all components
model = joblib.load('enhanced_accident_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
severity_mapping = joblib.load('severity_mapping.pkl')
feature_names = joblib.load('feature_names.pkl')

def predict_accident_severity(input_data):
    # 1. Add severe accident features
    input_data_enhanced = add_severe_accident_features(input_data)
    
    # 2. Ensure all expected columns are present
    for feature in feature_names:
        if feature not in input_data_enhanced.columns:
            input_data_enhanced[feature] = 0  # Add missing columns
    
    # 3. Reorder columns to match training
    input_data_enhanced = input_data_enhanced[feature_names]
    
    # 4. Scale numerical features
    input_data_enhanced[['speed_limit']] = scaler.transform(input_data_enhanced[['speed_limit']])
    
    # 5. Predict
    prediction_encoded = model.predict(input_data_enhanced)[0]
    prediction_proba = model.predict_proba(input_data_enhanced)[0]
    
    # 6. Convert to readable format
    prediction_readable = severity_mapping[prediction_encoded]
    
    return prediction_readable, prediction_proba, prediction_encoded

# Quick test to make sure everything loads correctly
try:
    # Test loading
    test_model = joblib.load('enhanced_accident_model.pkl')
    test_scaler = joblib.load('scaler.pkl')
    test_le = joblib.load('label_encoder.pkl')
    test_mapping = joblib.load('severity_mapping.pkl')
    test_features = joblib.load('feature_names.pkl')
    
    print("All files load successfully!")
    print(f"Model type: {type(test_model)}")
    print(f"Features: {len(test_features)} columns")
    print(f"Severity mapping: {test_mapping}")
    
except Exception as e:
    print(f"Error loading files: {e}")