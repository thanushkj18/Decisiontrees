import streamlit as st
import pickle
import numpy as np
import os

# Load model
model_path = os.path.join("model", "decision_tree.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🎓 Student Result Prediction")

# Form inputs
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.selectbox("Part Time Job", ["No", "Yes"])
absence_days = st.number_input("Absence Days", min_value=0, step=1)
extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
weekly_study = st.number_input("Weekly Self Study Hours", min_value=0, step=1)
career_aspiration = st.number_input("Career Aspiration (Encoded)", min_value=0, step=1)

# Convert inputs to numeric values (same encoding as training)
gender_val = 0 if gender == "Male" else 1
job_val = 0 if part_time_job == "No" else 1
extra_val = 0 if extracurricular == "No" else 1

if st.button("Predict"):
    features = np.array([[gender_val, job_val, absence_days,
                          extra_val, weekly_study, career_aspiration]])
    prediction = model.predict(features)[0]
    st.success(f"✅ Predicted Result: {prediction}")
