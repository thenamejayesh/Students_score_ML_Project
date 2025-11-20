 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional

# paths (use the inbuilt files you uploaded)
DEFAULT_MODEL = "Student_model (3).pkl" 
DEFAULT_CSV = "student_scores (1).csv"

st.set_page_config(page_title="Score Predictor", layout="centered")

# Helper utilities
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_estimator_from_loaded(obj):
    """
    If the pickle is a dict like {'model':..., 'encoder':...} return obj['model'].
    Otherwise return obj itself.
    """
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

def infer_expected_features(estimator) -> Optional[int]:
    """Try to infer number of features model expects; return int or None."""
    try:
        n = getattr(estimator, "n_features_in_", None)
        if n is not None:
            return int(n)
    except Exception:
        pass
    try:
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            arr = np.array(coef)
            if arr.ndim == 1:
                return arr.shape[0]
            elif arr.ndim == 2:
                return arr.shape[1]
    except Exception:
        pass
    return None

# Load default CSV (for showing sample and advice)
default_df = None
if os.path.exists(DEFAULT_CSV):
    try:
        default_df = pd.read_csv(DEFAULT_CSV)
    except Exception:
        default_df = None

# Sidebar uploads
st.sidebar.header("Files (optional)")
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"])
uploaded_csv = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

model_obj = None
if uploaded_model is not None:
    try:
        model_obj = pickle.load(uploaded_model)
        st.sidebar.success("Uploaded model loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not load uploaded model: {e}")
else:
    if os.path.exists(DEFAULT_MODEL):
        try:
            model_obj = load_model(DEFAULT_MODEL)
            st.sidebar.info(f"Loaded model from {DEFAULT_MODEL}")
        except Exception as e:
            st.sidebar.warning(f"Could not load default model: {e}")
    else:
        st.sidebar.info("No model found at default path.")

# CSV handling for dataset preview & sanity
if uploaded_csv is not None:
    try:
        default_df = pd.read_csv(uploaded_csv)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")

# Show sample dataset info
st.title("Score Predictor (using your dataset & model)")
if default_df is not None:
    st.markdown("**Dataset preview (first 5 rows)**")
    st.dataframe(default_df.head(5))
    st.write(f"Columns found: {default_df.columns.tolist()}")
else:
    st.info("No dataset preview available. Place your CSV at `/mnt/data/student_scores (1).csv` or upload it in the sidebar.")

# Input fields (based on CSV features)
st.markdown("---")
st.header("Enter features for prediction")

# Default values (sensible guesses)
hours = st.number_input("Hours_Studied", min_value=0.0, value=5.0, step=0.5, format="%.2f")
attendance = st.number_input("Attendance (percentage or count)", min_value=0.0, value=85.0, step=1.0, format="%.2f")
assignments = st.number_input("Assignments_Submitted", min_value=0, value=8, step=1)

# Build input dataframe
input_df = pd.DataFrame([{
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Assignments_Submitted": assignments
}])

st.markdown("### Input preview")
st.dataframe(input_df)

# If model missing, show helpful guidance and stop
if model_obj is None:
    st.error("No model loaded. Upload a .pkl model in the sidebar or place it at /mnt/data/Student_model (3).pkl")
    st.stop()

# Extract estimator if pickle stores extra objects
estimator = get_estimator_from_loaded(model_obj)

# Try to infer expected input feature count
expected_n = infer_expected_features(estimator)
if expected_n is not None:
    st.info(f"Model appears to expect **{expected_n}** input features.")
else:
    st.info("Model expected feature count could not be determined automatically.")

# Decide what to send to the model
# Most likely: model expects 3 features (Hours_Studied, Attendance, Assignments_Submitted)
final_X = None
if expected_n is None or expected_n == 3:
    # send the three numeric columns
    final_X = input_df[["Hours_Studied", "Attendance", "Assignments_Submitted"]].values
    st.write("Sending columns: Hours_Studied, Attendance, Assignments_Submitted")
elif expected_n == 4:
    # if model expects 4 features, try adding a placeholder for Score or a zero column (rare)
    # but safer to let user choose columns
    st.warning("Model expects 4 features while we have 3 inputs. Please choose which columns to send or upload a model trained on these 3 features.")
    cols = st.multiselect("Choose columns to send to model (order matters)", options=input_df.columns.tolist(), default=input_df.columns.tolist())
    if len(cols) == 0:
        st.error("No columns selected.")
    else:
        final_X = input_df[cols].values
else:
    # unexpected feature count: let user choose columns to send
    st.warning("Model expected feature count is unusual. Choose which columns to send (order matters).")
    cols = st.multiselect("Choose columns to send to model (order matters)", options=input_df.columns.tolist(), default=input_df.columns.tolist())
    if len(cols) == 0:
        st.error("No columns selected.")
    else:
        final_X = input_df[cols].values

# Predict button
if st.button("Predict"):
    if final_X is None:
        st.error("No input prepared for prediction.")
    else:
        try:
            pred = estimator.predict(final_X)
            # If array-like, take first
            pred_val = pred[0] if hasattr(pred, "__len__") else pred
            st.success(f"Predicted Score: **{pred_val}**")
            st.write("Raw model output:", pred)
            st.write("Final matrix shape sent to model:", final_X.shape)
        except Exception as e:
            st.error("Prediction failed. See details below.")
            st.exception(e)

# Footer / tips
st.markdown("---")
st.write("Tips:")
st.write("- If you trained the model on the 3 numeric features (Hours_Studied, Attendance, Assignments_Submitted) then this app will work as-is.")
st.write("- If your model was trained with a different preprocessing pipeline (scaling, encoders, feature selection), best practice is to save the full pipeline (or save a dict with 'model' + any preprocessing objects) and upload it here.")
