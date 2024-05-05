import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.image('https://innomatics.in/wp-content/uploads/2023/01/innomatics-footer-logo.png')

# Function to preprocess input data
def preprocess_input(data):
    # You can add any preprocessing steps here if required
    return data

# Function to predict failure
def predict_failure(input_data):
    # Load your trained model here using pickle
    with open("failuree.pkl", 'rb') as model_file:
        print("Loading the model...")
        model = pickle.load(model_file)
        print("Model loaded successfully.")

    # Dummy prediction for demonstration
    prediction = model.predict_proba(input_data)[:, 1]  # Assuming model returns probabilities

    if prediction > 0.7:
        return 'Ready to Fail'
    elif prediction == 1:
        return 'Failed'
    else:
        return 'Not Failed'

# Streamlit UI
st.title('Machine Failure Prediction App')

# Sidebar for input parameters
st.sidebar.header('Input Parameters')

# Input fields
type_options = [0, 1, 2]  # Define options for 'Type' (assuming it's already encoded numerically)
selected_type = st.sidebar.selectbox('Type', type_options)
air_temperature = st.sidebar.number_input('Air temperature [K]')
process_temperature = st.sidebar.number_input('Process temperature [K]')
rotational_speed = st.sidebar.number_input('Rotational speed [rpm]')
torque = st.sidebar.number_input('Torque [Nm]')
tool_wear = st.sidebar.number_input('Tool wear [min]')

input_df = pd.DataFrame({
    'Type': [selected_type],
    'Air temperature [K]': [air_temperature],
    'Process temperature [K]': [process_temperature],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
})

# Preprocess input data
input_data = preprocess_input(input_df)

# Button to trigger prediction
if st.sidebar.button('Predict'):
    prediction = predict_failure(input_data)
    st.write('Failure Prediction:', prediction)

