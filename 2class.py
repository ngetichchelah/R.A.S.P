import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="RASP🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load all components
@st.cache_resource
def load_models():
    try:
        assets = joblib.load('accident_severity_model.pkl')
        model = assets['model']
        scaler = assets['scaler']
        feature_names = assets['feature_names']
        performance_metrics = assets['performance_metrics']
        model_name = assets['model_name']
        
        # Create severity mapping for binary classification
        severity_mapping = {0: 'Severe', 1: 'Slight'}
        
        return model, scaler, feature_names, performance_metrics, model_name, severity_mapping
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Load models
model, scaler, feature_names, performance_metrics, model_name, severity_mapping = load_models()

st.title("Accident Severity Prediction App")
st.write("Enter accident details to predict severity (Slight or Severe):")

# Create input form based on available feature names
col1, col2, col3 = st.columns(3)

input_dict = {}

with col1:
    st.subheader("Location & Environment")
    
    if 'speed_limit' in feature_names:
        input_dict['speed_limit'] = st.number_input("Speed Limit", min_value=0, max_value=120, value=50)
    
    if 'urban_or_rural_area' in feature_names:
        input_dict['urban_or_rural_area'] = st.selectbox("Urban or Rural Area", [1, 2, 3], 
                                      format_func=lambda x: "Urban" if x == 1 else "Rural" if x == 2 else "Unallocated")
    
    if 'light_conditions' in feature_names:
        input_dict['light_conditions'] = st.selectbox("Light Conditions", [1, 4, 5, 6, 7], 
                                   format_func=lambda x: {
                                       1: "Daylight", 
                                       4: "Darkness - lights lit", 
                                       5: "Darkness - lights unlit", 
                                       6: "Darkness - no lighting", 
                                       7: "Darkness - lighting unknown"
                                   }[x])
    
    if 'weather_conditions' in feature_names:
        input_dict['weather_conditions'] = st.selectbox("Weather Conditions", [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     format_func=lambda x: {
                                         1: "Fine no high winds", 
                                         2: "Raining no high winds", 
                                         3: "Snowing no high winds",
                                         4: "Fine + high winds", 
                                         5: "Raining + high winds", 
                                         6: "Snowing + high winds",
                                         7: "Fog or mist", 
                                         8: "Other", 
                                         9: "Unknown"
                                     }[x])

with col2:
    st.subheader("Road & Junction Details")
    
    if 'first_road_class' in feature_names:
        input_dict['first_road_class'] = st.selectbox("First Road Class", [1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: {
                                       1: "Motorway", 
                                       2: "A(M)", 
                                       3: "A", 
                                       4: "B", 
                                       5: "C", 
                                       6: "Unclassified"
                                   }[x])
    
    if 'road_type' in feature_names:
        input_dict['road_type'] = st.selectbox("Road Type", [1, 2, 3, 6, 7, 9],
                            format_func=lambda x: {
                                1: "Roundabout", 
                                2: "One way street", 
                                3: "Dual carriageway",
                                6: "Single carriageway", 
                                7: "Slip road", 
                                9: "Unknown"
                            }[x])
    
    if 'junction_control' in feature_names:
        input_dict['junction_control'] = st.selectbox("Junction Control", [1, 2, 3, 4, 9],
                                   format_func=lambda x: {
                                       1: "Not at junction or within 20 metres", 
                                       2: "Authorised person", 
                                       3: "Auto traffic signal",
                                       4: "Stop sign", 
                                       9: "Give way or uncontrolled"
                                   }[x])
    
    if 'pedestrian_crossing_human_control' in feature_names:
        input_dict['pedestrian_crossing_human_control'] = st.selectbox("Pedestrian Crossing Human Control", [0, 1, 2, 9],
                                                    format_func=lambda x: {
                                                        0: "None", 
                                                        1: "Control by school crossing patrol", 
                                                        2: "Control by other authorised person",
                                                        9: "Unknown"
                                                    }[x])

with col3:
    st.subheader("Time & Conditions")
    
    if 'is_rush_hour' in feature_names:
        input_dict['is_rush_hour'] = st.selectbox("Is Rush Hour", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    if 'is_night' in feature_names:
        input_dict['is_night'] = st.selectbox("Is Night", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    #if 'is_weekend' in feature_names:
       #input_dict['is_weekend'] = st.selectbox("Is Weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    if 'day_of_week' in feature_names:
        input_dict['day_of_week'] = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7],
                              format_func=lambda x: {
                                  1: "Sunday", 
                                  2: "Monday", 
                                  3: "Tuesday", 
                                  4: "Wednesday", 
                                  5: "Thursday", 
                                  6: "Friday", 
                                  7: "Saturday"
                              }[x])
    
    if 'month' in feature_names:
        input_dict['month'] = st.slider("Month", 1, 12, 6)

# # Add location inputs if they're in features
# if any('latitude' in f for f in feature_names):
#     st.subheader("Geographic Location")
#     lat_col, lon_col = st.columns(2)
#     with lat_col:
#         input_dict['latitude'] = st.number_input("Latitude", value=51.5074, format="%.6f")
#     with lon_col:
#         input_dict['longitude'] = st.number_input("Longitude", value=-0.1278, format="%.6f")

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Preprocessing function
def preprocess_input(input_df, scaler, feature_names):
    # Add missing columns with default 0
    missing_cols = [col for col in feature_names if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0
    
    # Reorder columns to match training
    input_df = input_df[feature_names]
    
    # Scale numeric features
    if scaler:
        numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    return input_df

# Predict button
if st.button("Predict Accident Severity", type="primary"):
    if model is not None:
        # Preprocess input
        input_df_processed = preprocess_input(input_df.copy(), scaler, feature_names)
        
        # Make prediction
        prediction = model.predict(input_df_processed)[0]
        prediction_proba = model.predict_proba(input_df_processed)[0]
        
        # Display results
        st.subheader("Prediction Results:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success(f"Predicted: {severity_mapping[prediction]}")
            else:
                st.error(f"Predicted: {severity_mapping[prediction]}")
        
        # with col2:
        #     confidence = max(prediction_proba) * 100
        #     st.metric("Confidence", f"{confidence:.1f}%")
        
        # Show probability breakdown as percentages only (no visualization)
        st.subheader("Probability Breakdown")
        
        # Create a clean table with percentages
        prob_data = []
        for severity, prob in zip(list(severity_mapping.values()), prediction_proba):
            prob_data.append({
                "Severity Type": severity,
                "Probability": f"{prob * 100:.1f}%"
            })
        
        # Display as a clean table
        prob_df = pd.DataFrame(prob_data)
        st.table(prob_df)
        
        # Optional: Add some color coding
        severe_prob = prediction_proba[0] * 100
        slight_prob = prediction_proba[1] * 100
        
        st.write("**Detailed Analysis:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if severe_prob > 60:
                st.error(f"High probability of Severe accident: {severe_prob:.1f}%")
            elif severe_prob > 30:
                st.warning(f"Moderate probability of Severe accident: {severe_prob:.1f}%")
            else:
                st.info(f"Low probability of Severe accident: {severe_prob:.1f}%")
        
        with col2:
            if slight_prob > 60:
                st.success(f"High probability of Slight accident: {slight_prob:.1f}%")
            elif slight_prob > 30:
                st.info(f"Moderate probability of Slight accident: {slight_prob:.1f}%")
            else:
                st.warning(f"Low probability of Slight accident: {slight_prob:.1f}%")
    
    else:
        st.error("Model not loaded properly. Please check the model files.")

# Model information in sidebar
st.sidebar.header("Model Information")
if model is not None:
    st.sidebar.success("RASP!")
        
# Footer
st.markdown("---")
st.markdown("**Accident Severity Prediction Model** | Built with Streamlit")