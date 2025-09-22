import streamlit as st
import pandas as pd
import joblib


model_path = "W:/Pycharm_Projects/Sprints/models/final_pipeline.pkl"
pipeline = joblib.load(model_path)

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details to predict the likelihood of heart disease:")

# Input fields with realistic ranges & labels
age = st.number_input("Age (years)", min_value=20, max_value=100, value=40)

# Sex (0 = Female, 1 = Male)
sex_option = st.radio("Sex", ["Male", "Female"])
sex = 1 if sex_option == "Male" else 0

# Chest Pain Type
cp_dict = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp_option = st.selectbox("Chest Pain Type", list(cp_dict.keys()))
cp = cp_dict[cp_option]

# Resting BP
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

# Cholesterol
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

# Fasting Blood Sugar
fbs_option = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
fbs = 1 if fbs_option == "Yes" else 0

# Resting ECG
restecg_dict = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Possible/Definite LVH": 2
}
restecg_option = st.selectbox("Resting ECG", list(restecg_dict.keys()))
restecg = restecg_dict[restecg_option]

# Max Heart Rate
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)

# Exercise Induced Angina
exang_option = st.radio("Exercise Induced Angina", ["Yes", "No"])
exang = 1 if exang_option == "Yes" else 0

# Oldpeak
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

# Slope
slope_dict = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope_option = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_dict.keys()))
slope = slope_dict[slope_option]

# Number of major vessels
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

# Thalassemia
thal_dict = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}
thal_option = st.selectbox("Thalassemia", list(thal_dict.keys()))
thal = thal_dict[thal_option]

# Make prediction
if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal"
    ])

    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f" Result: Patient is at risk of Heart Disease.\n\nProbability = {proba:.2f}")
    else:
        st.success(f" Result: Patient is Healthy.\n\nProbability = {proba:.2f}")
