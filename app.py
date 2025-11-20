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
    st.subheader("📘 Datase
