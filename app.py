import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ----------------- CONFIG -----------------
DEFAULT_MODEL_PATH = "/mnt/data/Student_model (1).pkl"
DEFAULT_CSV_PATH = "/mnt/data/student_scores (1).csv"
# ------------------------------------------

st.set_page_config(page_title="Student Prediction App", layout="centered")
st.title("Student Prediction App")
st.write("Load model + training CSV or upload them below. Enter all features and click Predict.")

# ---------- Helpers ----------
@st.cache_data
def load_pickle_model(path):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None, f"CSV file not found at: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {e}"

def build_label_encoder(series):
    le = LabelEncoder()
    vals = series.dropna().astype(str).unique().tolist()
    if len(vals) == 0:
        return None
    le.fit(vals)
    return le

# ---------- File inputs (optional override) ----------
st.sidebar.header("Model & Data (optional upload)")
uploaded_model = st.sidebar.file_uploader("Upload pickled model (.pkl)", type=["pkl", "joblib"])
uploaded_csv = st.sidebar.file_uploader("Upload training CSV (used to build encoder)", type=["csv"])

use_default_paths = st.sidebar.checkbox("Use default files in /mnt/data if available", value=True)

# Load model
model = None
model_err = None
if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
    except Exception as e:
        model_err = f"Uploaded model load error: {e}"
elif use_default_paths and os.path.exists(DEFAULT_MODEL_PATH):
    model, model_err = load_pickle_model(DEFAULT_MODEL_PATH)
else:
    model_err = "No model provided. Upload a .pkl or enable default path in sidebar."

# Load CSV (to build encoder)
train_df = None
csv_err = None
if uploaded_csv is not None:
    try:
        uploaded_csv.seek(0)
        train_df = pd.read_csv(uploaded_csv)
    except Exception as e:
        csv_err = f"Uploaded CSV load error: {e}"
elif use_default_paths and os.path.exists(DEFAULT_CSV_PATH):
    train_df, csv_err = load_csv(DEFAULT_CSV_PATH)
else:
    csv_err = "No CSV provided. Upload training CSV or enable default path in sidebar."

# Show any loading errors
if model_err:
    st.error(model_err)
if csv_err:
    st.warning(csv_err)

if model is None:
    st.stop()

# Build encoder if department exists
le = None
dept_options = []
if train_df is not None and "department" in train_df.columns:
    try:
        le = build_label_encoder(train_df["department"])
        if le is not None:
            dept_options = list(le.classes_)
    except Exception as e:
        st.warning(f"Could not build label encoder from CSV: {e}")

# Determine default medians for numeric inputs if available
def safe_median(df, col, fallback):
    try:
        if df is not None and col in df.columns:
            return float(df[col].median())
    except Exception:
        pass
    return fallback

default_age = safe_median(train_df, "age", 25)
default_experience = safe_median(train_df, "experience", 1.0)
default_salary = safe_median(train_df, "salary", 30000.0)

# ---------- User input form ----------
st.subheader("Input features")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=200, value=int(default_age))
        experience = st.number_input("Experience (years)", min_value=0.0, max_value=100.0,
                                     value=float(default_experience), step=0.5)
    with col2:
        salary = st.number_input("Salary", min_value=0.0, value=float(default_salary), step=100.0, format="%.2f")
        if dept_options:
            department = st.selectbox("Department", options=dept_options)
        else:
            department = st.text_input("Department (type a value)")
    submitted = st.form_submit_button("Predict")

if not submitted:
    st.info("Fill inputs and press Predict.")
    st.stop()

# Validate inputs
if any(v is None or (isinstance(v, str) and v.strip() == "") for v in [age, experience, salary, department]):
    st.error("All features are required. Please fill every field.")
    st.stop()

# Prepare input vector
try:
    if le is not None:
        # if unseen label, map to -1
        dept_str = str(department)
        if dept_str in le.classes_:
            dept_enc = int(le.transform([dept_str])[0])
        else:
            dept_enc = -1
    else:
        # try numeric conversion; if not numeric, keep as string (some pipelines handle it)
        try:
            dept_enc = float(department)
        except Exception:
            dept_enc = str(department)

    X = np.array([[age, experience, salary, dept_enc]], dtype=object)

    # Try cast to float if possible (many models expect numeric array)
    try:
        X = X.astype(float)
    except Exception:
        pass

    # Predict
    pred = model.predict(X)
    pred_val = pred[0] if hasattr(pred, "__len__") else pred

    st.subheader("Prediction")
    st.success(f"Predicted value: {pred_val}")

    # If classification probability available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            class_labels = getattr(model, "classes_", None)
            if class_labels is not None:
                proba_df = pd.DataFrame(proba, columns=[str(c) for c in class_labels])
            else:
                proba_df = pd.DataFrame(proba, columns=[f"prob_{i}" for i in range(proba.shape[1])])
            st.subheader("Prediction probabilities")
            st.dataframe(proba_df)
        except Exception as e:
            st.info(f"predict_proba failed: {e}")

    # Optionally show model info
    st.markdown("**Model info:**")
    try:
        st.write(type(model))
        if hasattr(model, "score"):
            st.write("Model has `score` method.")
    except Exception:
        pass

except Exception as err:
    st.error(f"Prediction failed: {err}")
