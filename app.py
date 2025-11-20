import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------
# Paths for inbuilt files
# --------------------------------------------------
DEFAULT_MODEL = "Student_model (3).pkl"
DEFAULT_CSV = "student_scores (1).csv"

st.set_page_config(page_title="Score Predictor", layout="centered")

# --------------------------------------------------
# Helper
# --------------------------------------------------
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_model(obj):
    """If pickle contains {'model':..., ...} return obj['model'] else return original."""
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

# --------------------------------------------------
# Load CSV (preview only)
# --------------------------------------------------
df = None
if os.path.exists(DEFAULT_CSV):
    try:
        df = pd.read_csv(DEFAULT_CSV)
    except:
        df = None

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model_obj = None
if os.path.exists(DEFAULT_MODEL):
    try:
        model_obj = load_model(DEFAULT_MODEL)
    except Exception as e:
        st.error(f"Could not load model: {e}")
else:
    st.error("Model file is missing! Upload Student_model (3).pkl inside your project folder.")
    st.stop()

model = extract_model(model_obj)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎯 Student Score Predictor")

if df is not None:
    st.subheader("📘 Dataset Preview")
    st.dataframe(df.head(5))
else:
    st.info("Could not load dataset preview. Ensure student_scores (1).csv exists.")

st.markdown("---")
st.header("Enter Details for Prediction")

# --------------------------------------------------
# Only 3 Features (As You Requested)
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    Hours_Studied = st.number_input("Hours Studied", min_value=0.0, value=5.0, step=0.5)

with col2:
    Attendance = st.number_input("Attendance (%)", min_value=0.0, value=85.0, step=1.0)

with col3:
    Assignments = st.number_input("Assignments Submitted", min_value=0, value=8, step=1)

# Prepare 3-feature input
input_df = pd.DataFrame([{
    "Hours_Studied": Hours_Studied,
    "Attendance": Attendance,
    "Assignments_Submitted": Assignments
}])

st.subheader("🔍 Input Preview")
st.dataframe(input_df)

# --------------------------------------------------
# Predict Button
# --------------------------------------------------
if st.button("Predict Score"):
    try:
        X = input_df.values  # always 3-column input
        pred = model.predict(X)
        result = pred[0]

        st.success(f"📊 Predicted Score: **{result}**")

    except Exception as e:
        st.error("❌ Prediction failed!")
        st.exception(e)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("This app uses exactly **3 input features** and automatically loads the inbuilt model & dataset.")
