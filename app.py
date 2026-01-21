import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("heart.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter all patient details below to predict heart disease risk.")

st.divider()

# ---------- INPUTS (EXACT 13 FEATURES) ----------

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=45
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3],
    help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"
)

trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Serum Cholesterol (chol) in mg/dl",
    min_value=50,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    options=[0, 1],
    format_func=lambda x: "False" if x == 0 else "True"
)

restecg = st.selectbox(
    "Resting ECG Results (restecg)",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "ST Depression (oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope of the Peak Exercise ST Segment (slope)",
    options=[0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy (ca)",
    options=[0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia (thal)",
    options=[0, 1, 2, 3]
)

st.divider()

# ---------- PREDICTION ----------

if st.button("🔍 Predict Heart Disease"):
    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
        st.write("👉 Recommendation: Please consult a cardiologist immediately.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write("👉 Recommendation: Maintain a healthy lifestyle.")

st.caption("Model: Random Forest | Deployed using Streamlit")
