import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "Student_model (1).pkl"
CSV_PATH = "student_scores (1).csv"

st.set_page_config(page_title="Prediction App", layout="centered")
st.title("Prediction App (Local Files)")

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success(f"Loaded model: {MODEL_PATH}")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
    st.success(f"Loaded CSV: {CSV_PATH} (rows: {len(df)})")
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

# Label encoder
try:
    le = LabelEncoder()
    le.fit(df["department"].astype(str))
    dept_options = list(le.classes_)
except:
    st.error("CSV must contain a 'department' column.")
    st.stop()

# Medians for defaults
def med(c, fallback): 
    return float(df[c].median()) if c in df.columns else fallback

default_age = med("age", 25)
default_exp = med("experience", 1.0)
default_sal = med("salary", 30000.0)

st.subheader("Enter features")
with st.form("f"):
    age = st.number_input("Age", min_value=0, max_value=200, value=int(default_age))
    exp = st.number_input("Experience", min_value=0.0, max_value=100.0, value=float(default_exp), step=0.5)
    salary = st.number_input("Salary", min_value=0.0, value=float(default_sal), step=100.0)
    department = st.selectbox("Department", dept_options)

    submit = st.form_submit_button("Predict")

if submit:
    dept_enc = le.transform([department])[0]

    X = np.array([[age, exp, salary, dept_enc]], dtype=float)

    try:
        pred = model.predict(X)
        st.success(f"Prediction: {pred[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
