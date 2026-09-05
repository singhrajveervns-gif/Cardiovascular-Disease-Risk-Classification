import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Cardiovascular Disease Risk", layout="centered")

st.title("Cardiovascular Disease Risk Predictor")
st.caption(
    "Decision-support demo based on routine checkup indicators. "
    "This is a portfolio project, not a medical diagnostic tool."
)

@st.cache_resource
def load_model():
    return joblib.load("models/cardio_final_model.pkl")

model = load_model()

st.sidebar.header("Patient checkup details")

age = st.sidebar.slider("Age (years)", 20, 90, 50)
gender = st.sidebar.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
height = st.sidebar.slider("Height (cm)", 130, 210, 165)
weight = st.sidebar.slider("Weight (kg)", 35, 160, 70)
ap_hi = st.sidebar.slider("Systolic BP (ap_hi)", 80, 220, 120)
ap_lo = st.sidebar.slider("Diastolic BP (ap_lo)", 50, 150, 80)
cholesterol = st.sidebar.selectbox("Cholesterol level", options=[1, 2, 3],
                                    format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
gluc = st.sidebar.selectbox("Glucose level", options=[1, 2, 3],
                             format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
smoke = st.sidebar.selectbox("Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alco = st.sidebar.selectbox("Consumes alcohol", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
active = st.sidebar.selectbox("Physically active", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Engineered features -- must match the notebook's Feature Engineering step exactly
bmi = round(weight / ((height / 100) ** 2), 2)
pulse_pressure = ap_hi - ap_lo

st.subheader("Derived values")
col1, col2 = st.columns(2)
col1.metric("BMI", bmi)
col2.metric("Pulse pressure", pulse_pressure)

if ap_lo >= ap_hi:
    st.warning("Diastolic BP should be lower than systolic BP. Adjust the sliders before predicting.")

input_row = pd.DataFrame([{
    "age": age, "gender": gender, "height": height, "weight": weight,
    "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol, "gluc": gluc,
    "smoke": smoke, "alco": alco, "active": active,
    "bmi": bmi, "pulse_pressure": pulse_pressure,
}])

# Ensure column order matches the fitted pipeline's expected input
expected_cols = model.named_steps["prep"].transformers_[0][2]
input_row = input_row[expected_cols]

if st.button("Predict cardiovascular disease risk"):
    proba = model.predict_proba(input_row)[0, 1]
    prediction = "At risk" if proba >= 0.5 else "Lower risk"

    st.subheader("Result")
    st.metric("Predicted probability of cardiovascular disease", f"{proba:.1%}")
    if prediction == "At risk":
        st.error(f"Prediction: {prediction}")
    else:
        st.success(f"Prediction: {prediction}")

    st.caption(
        "This estimate reflects statistical association learned from historical checkup data. "
        "It is not a diagnosis and does not replace clinical judgment or testing."
    )
