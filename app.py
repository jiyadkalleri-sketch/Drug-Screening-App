import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Set up the Streamlit app interface
st.title('Simulated Drug Screening Model')
st.write('Enter the molecular descriptors and feature values below to predict whether the compound is active or inactive.')

# Input fields for the 6 specific features used in Option C[cite: 2]
molecular_weight = st.number_input('Molecular Weight', value=0.0)
binding_affinity = st.number_input('Binding Affinity', value=0.0)
solubility_score = st.number_input('Solubility Score', value=0.0)
toxicity_score = st.number_input('Toxicity Score', value=0.0)
feature_5 = st.number_input('Feature 5', value=0.0)
feature_6 = st.number_input('Feature 6', value=0.0)

# Prediction logic
if st.button('Predict'):
    # Combine user inputs into an array matching the training feature order
    input_data = np.array([[molecular_weight, binding_affinity, solubility_score, toxicity_score, feature_5, feature_6]])
    
    # Scale input data and generate prediction
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    
    # Display the result with appropriate UI feedback
    if result[0] == 1:
        st.success('Prediction: Active Compound (Class 1)')
    else:
        st.warning('Prediction: Inactive Compound (Class 0)')
