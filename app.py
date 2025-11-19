import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---- CONFIG: exact expected default paths ----
MODEL_PATH = "/mnt/data/Student_model (1).pkl"
CSV_PATH = "/mnt/data/student_scores (1).csv"
# ----------------------------------------------

st.set_page_config(page_title="Student Prediction (auto-load /mnt/data)", layout="centered")
st.title("Student Prediction (auto-load /mnt/data)")

st.markdown("This version **automatically** attempts to load model & CSV from `/mnt/data`.")
st.subheader("Contents of /mnt/data")
try:
    files = os.listdir("/mnt/data")
    if len(files) == 0:
        st.warning("/mnt/data exists but is empty.")
    else:
        st.write(files)
except Exception as e:
    st.error(f"Could not list /mnt/data: {e}")
    files = []

# Try to load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        st.success(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        st.error(f"Failed to load model at {MODEL_PATH}: {e}")
else:
    st.error(f"Model file not found at {MODEL_PATH}")

# Try to load CSV
train_df = None
if os.path.exists(CSV_PATH):
    try:
        train_df = pd.read_csv(CSV_PATH)
        st.success(f"Loaded CSV from {CSV_PATH} (rows: {len(train_df)})")
    except Exception as e:
        st.error(f"Failed to load CSV at {CSV_PATH}: {e}")
else:
    st.error(f"CSV file not found at {CSV_PATH}")

if model is None or train_df is None:
    st.stop()

# Build encoder if possible
le = None
dept_options = []
if "department" in train_df.columns:
    try:
        le = LabelEncoder()
        le.fit(train_df["department"].astype(str).unique().tolist())
        dept_options = list(le.classes_)
    except Exception as e:
        st.warning(f"Could not build encoder from CSV: {e}")

# Default medians
def med(col, fallback):
    try:
        return float(train_df[col].median()) if col in train_df.columns else fallback
    except Exception:
        return fallback

default_age = med("age", 25)
default_experience = med("experience", 1.0)
default_salary = med("salary", 30000.0)

st.subheader("Enter inputs")
with st.form("f"):
    age = st.number_input("Age", min_value=0, max_value=200, value=int(default_age))
    experience = st.number_input("Experience (years)", min_value=0.0, max_value=100.0, value=float(default_experience), step=0.5)
    salary = st.number_input("Salary", min_value=0.0, value=float(default_salary), step=100.0, format="%.2f")
    if dept_options:
        department = st.selectbox("Department", dept_options)
    else:
        department = st.text_input("Department")
    ok = st.form_submit_button("Predict")

if not ok:
    st.info("Fill inputs and click Predict.")
    st.stop()

# Prepare X
try:
    if le is not None:
        dept_str = str(department)
        dept_enc = int(le.transform([dept_str])[0]) if dept_str in le.classes_ else -1
    else:
        try:
            dept_enc = float(department)
        except Exception:
            dept_enc = str(department)

    X = np.array([[age, experience, salary, dept_enc]], dtype=object)
    try:
        X = X.astype(float)
    except Exception:
        pass

    pred = model.predict(X)
    val = pred[0] if hasattr(pred, "__len__") else pred
    st.success(f"Predicted value: {val}")

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            cls = getattr(model, "classes_", None)
            dfp = pd.DataFrame(proba, columns=[str(c) for c in cls]) if cls is not None else pd.DataFrame(proba)
            st.dataframe(dfp)
        except Exception as e:
            st.info(f"predict_proba error: {e}")

except Exception as e:
    st.error(f"Prediction failed: {e}")
