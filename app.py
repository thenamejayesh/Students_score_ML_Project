import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_PATH = "/mnt/data/Student_model (3).pkl"
DEFAULT_DATA_PATH = "/mnt/data/student_scores (1).csv"
EXPECTED_FEATURES = ["age", "experience", "salary", "department"]

st.set_page_config(page_title="Student Score Predictor", layout="centered")

# -----------------------------
# Helper Functions
# -----------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_label_encode(values, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(values.astype(str))
    else:
        try:
            encoded = encoder.transform(values.astype(str))
        except:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(values.astype(str))
    return encoded, encoder

def build_input_df(age, experience, salary, department, encoder=None):
    df = pd.DataFrame([{
        "age": age,
        "experience": experience,
        "salary": salary,
        "department": department
    }])
    encoded_dept, enc = safe_label_encode(df["department"], encoder)
    df["department"] = encoded_dept
    return df[EXPECTED_FEATURES], enc

def reorder_features(df):
    return df[EXPECTED_FEATURES]

# -----------------------------
# Load Defaults
# -----------------------------
@st.cache_data
def load_defaults():
    model = None
    data = None

    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            model = load_pickle(DEFAULT_MODEL_PATH)
        except:
            model = None

    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            data = pd.read_csv(DEFAULT_DATA_PATH)
        except:
            data = None

    return model, data

default_model, default_data = load_defaults()

# -----------------------------
# Sidebar Upload
# -----------------------------
st.sidebar.header("File Upload")

uploaded_model = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])
uploaded_csv = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

model = None
data_df = None
encoder = None

# Load model
if uploaded_model:
    try:
        model = pickle.load(uploaded_model)
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded model: {e}")
else:
    model = default_model
    if model:
        st.sidebar.info("Loaded model from system path.")

# Load CSV
if uploaded_csv:
    try:
        data_df = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
else:
    data_df = default_data

# Build department options
if data_df is not None and "department" in data_df.columns:
    dept_options = sorted(data_df["department"].astype(str).unique().tolist())
else:
    dept_options = ["Sales", "HR", "Finance", "IT", "Marketing"]

# Fit encoder from CSV
if data_df is not None and "department" in data_df.columns:
    try:
        _, encoder = safe_label_encode(data_df["department"])
    except:
        encoder = None

# -----------------------------
# Main Interface
# -----------------------------
st.title("Student / Employee Score Predictor")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 0, 120, 22)
    salary = st.number_input("Salary", 0, 200000, 30000)
with col2:
    experience = st.number_input("Experience (Years)", 0.0, 50.0, 1.0)
    department = st.selectbox("Department", dept_options)

st.markdown("---")

if st.button("Predict"):
    if model is None:
        st.error("No model loaded!")
    else:
        try:
            input_df, used_encoder = build_input_df(
                age, experience, salary, department, encoder
            )
            input_df = reorder_features(input_df)

            estimator = model["model"] if isinstance(model, dict) and "model" in model else model

            pred = estimator.predict(input_df.values)
            pred_val = pred[0]

            st.success(f"Predicted Score: **{pred_val}**")
            st.write("Processed Input:")
            st.dataframe(input_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
