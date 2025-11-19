 # app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# ---------- Config ----------
DEFAULT_MODEL_PATH = "/mnt/data/Student_model (3).pkl"
DEFAULT_DATA_PATH = "/mnt/data/student_scores (1).csv"
EXPECTED_FEATURES = ["age", "experience", "salary", "department"]  # training features you mentioned

st.set_page_config(page_title="Student Score Predictor", layout="centered")

# ---------- Utility functions ----------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_label_encode(series, encoder=None):
    """
    If encoder provided (LabelEncoder instance), use it; otherwise fit a new encoder and return it.
    Returns (encoded_array, encoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series.astype(str))
    else:
        try:
            encoded = encoder.transform(series.astype(str))
        except Exception:
            # fallback: fit encoder to series (useful if encoder doesn't contain all categories)
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(series.astype(str))
    return encoded, encoder

def build_input_df(age, experience, salary, department, encoder=None):
    """
    Create a one-row DataFrame with expected features and encoded department.
    Returns df, encoder_used
    """
    df = pd.DataFrame([{
        "age": age,
        "experience": experience,
        "salary": salary,
        "department": department
    }])
    encoded_dept, encoder_used = safe_label_encode(df["department"], encoder)
    df["department"] = encoded_dept
    return df[EXPECTED_FEATURES], encoder_used

def reorder_features(df, expected):
    """
    Ensure df has columns in expected order. If missing columns raise ValueError.
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features required by model: {missing}")
    return df[expected]

# ---------- Load defaults if present ----------
@st.cache_data(show_spinner=False)
def load_defaults():
    default_model = None
    default_data = None
    model_loaded_from = None
    data_loaded_from = None

    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            default_model = load_pickle(DEFAULT_MODEL_PATH)
            model_loaded_from = DEFAULT_MODEL_PATH
        except Exception as e:
            default_model = None

    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            default_data = pd.read_csv(DEFAULT_DATA_PATH)
            data_loaded_from = DEFAULT_DATA_PATH
        except Exception:
            default_data = None

    return default_model, model_loaded_from, default_data, data_loaded_from

default_model, default_model_path, default_data, default_data_path = load_defaults()

# ---------- Sidebar: optional uploads and info ----------
st.sidebar.header("Files / Settings")

uploaded_model = st.sidebar.file_uploader("Upload model (.pkl) (optional)", type=["pkl"])
uploaded_csv = st.sidebar.file_uploader("Upload dataset CSV (optional) — used to get categories", type=["csv"])

model = None
encoder = None
data_df = None

# Load model (uploaded takes precedence)
if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.sidebar.success("Uploaded model loaded.")
    except Exception as e:
        st.sidebar.error(f"Uploaded model could not be loaded: {e}")
else:
    if default_model is not None:
        model = default_model
        st.sidebar.write(f"Loaded model from `{default_model_path}`")
    else:
        st.sidebar.info("No model loaded yet. Upload .pkl or place it at /mnt/data/Student_model (3).pkl")

# Load dataset for categories (uploaded takes precedence)
if uploaded_csv is not None:
    try:
        data_df = pd.read_csv(uploaded_csv)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Uploaded CSV could not be read: {e}")
else:
    if default_data is not None:
        data_df = default_data
        st.sidebar.write(f"Loaded CSV from `{default_data_path}`")
    else:
        st.sidebar.info("No CSV loaded. Upload CSV or place it at /mnt/data/student_scores (1).csv")

# Try to detect encoder saved inside model
if model is not None:
    # Common patterns: model might be a pipeline, a dict, or an sklearn estimator
    try:
        # If model is a dict-like with encoder saved
        if isinstance(model, dict) and "encoder" in model:
            encoder = model["encoder"]
        # If it's a custom object with attribute encoder
        elif hasattr(model, "encoder"):
            encoder = getattr(model, "encoder")
        # If it's a pipeline, try to find a LabelEncoder or similar in steps (best-effort)
        elif hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "classes_") or isinstance(step, LabelEncoder):
                    encoder = step
                    break
    except Exception:
        encoder = None

# If no encoder found but data_df available, fit encoder from its department column
if encoder is None and data_df is not None:
    if "department" in data_df.columns:
        try:
            _, encoder = safe_label_encode(data_df["department"])
            st.sidebar.write("Fitted LabelEncoder from dataset `department` column.")
        except Exception:
            encoder = None

# Provide department choices for UI, attempting to use data_df if available
dept_options = []
if data_df is not None and "department" in data_df.columns:
    dept_options = sorted(data_df["department"].astype(str).unique().tolist())
else:
    # sensible fallback options
    dept_options = ["Sales", "HR", "Finance", "IT", "Marketing", "Operations"]

# ---------- Main UI ----------
st.title("Student / Employee Score Predictor")
st.write("Enter all features below (age, experience, salary, department) and press **Predict**.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
    salary = st.number_input("Salary", min_value=0.0, value=30000.0, step=100.0, format="%.2f")
with col2:
    experience = st.number_input("Experience (years)", min_value=0.0, max_value=80.0, value=2.0, step=0.5)
    department = st.selectbox("Department", options=dept_options)

st.markdown("---")
st.write("Model & preprocessing status:")
if model is None:
    st.error("No model loaded. Upload a .pkl model file in the sidebar or place it at `/mnt/data/Student_model (3).pkl`.")
else:
    st.success("Model loaded.")

if encoder is None:
    st.warning("No label encoder available. Department will be fitted on-the-fly (may differ from model training).")
else:
    st.info("LabelEncoder is available for department encoding.")

# Predict button
if st.button("Predict"):
    if model is None:
        st.error("Cannot predict: no model loaded.")
    else:
        try:
            # Build input row and encode department consistently
            i
