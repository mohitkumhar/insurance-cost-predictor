import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model and preprocessing objects
model = load_model('model.h5')
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
ohe_sex = pickle.load(open('ohe_sex.pkl', 'rb'))
ohe_smoke = pickle.load(open('ohe_smoker.pkl', 'rb'))
le = pickle.load(open('label.pkl', 'rb'))

# Function to predict insurance cost
def predict_insurance(age, sex, bmi, children, smoker, region):
    # Check if the region is valid
    if region not in le.classes_:
        raise ValueError(f"Unknown region '{region}'. Valid options: {list(le.classes_)}")

    # Encode inputs
    sex_encoded = ohe_sex.transform(pd.DataFrame([[sex]], columns=['sex'])).toarray().flatten()
    smoker_encoded = ohe_smoke.transform(pd.DataFrame([[smoker]], columns=['smoker'])).toarray().flatten()
    region_encoded = le.transform([region])

    # Combine all inputs into one array
    input_data = np.hstack(([age, bmi, children], sex_encoded, smoker_encoded, region_encoded))

    # Scale the input data
    input_scaled = scaler_X.transform([input_data])

    # Make prediction
    prediction_scaled = model.predict(input_scaled)

    # Convert scaled prediction back to original value
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]


    return round(abs(prediction), 2)

# Streamlit UI
def main():
    st.title("Insurance Cost Prediction App 💼")
    st.write("Enter the details below to predict the insurance cost:")

    # User inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", le.classes_)

    # Predict button
    if st.button("Predict"):
        try:
            result = predict_insurance(age, sex, bmi, children, smoker, region)
            st.success(f"Predicted Insurance Cost: **${result}**")
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
