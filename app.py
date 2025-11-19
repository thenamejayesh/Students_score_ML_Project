import os
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ----------------- Config - change if your filenames differ -----------------
DEFAULT_MODEL_FILE = "Student_model (1).pkl"
DEFAULT_CSV_FILE = "student_scores (1).csv"
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Student Score Dashboard", layout="wide")
st.title("📊 Student Score Dashboard")
st.write("Load the model and training CSV, enter input features, then click **Predict Score**.")

# ---------- Sidebar: upload or use defaults ----------
st.sidebar.header("Model & Data")
use_defaults = st.sidebar.checkbox("Use default files from current folder if available", value=True)
uploaded_model = st.sidebar.file_uploader("Upload pickled model (.pkl)", type=["pkl", "joblib"])
uploaded_csv = st.sidebar.file_uploader("Upload training CSV (for encoder & stats)", type=["csv"])

# ---------- Load model ----------
@st.cache_data(show_spinner=False)
def load_model_from_path(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = None
model_load_error = None
if uploaded_model is not None:
    try:
        # uploaded_model is a BytesIO-like object
        uploaded_model.seek(0)
        model = pickle.load(uploaded_model)
    except Exception as e:
        model_load_error = f"Error loading uploaded model: {e}"
elif use_defaults and os.path.exists(DEFAULT_MODEL_FILE):
    try:
        model = load_model_from_path(DEFAULT_MODEL_FILE)
    except Exception as e:
        model_load_error = f"Error loading model file '{DEFAULT_MODEL_FILE}': {e}"
else:
    model_load_error = "No model provided. Upload a .pkl in the sidebar or place the default file in the app folder."

# ---------- Load CSV ----------
train_df = None
csv_load_error = None
@st.cache_data(show_spinner=False)
def load_csv_from_path(path):
    return pd.read_csv(path)

if uploaded_csv is not None:
    try:
        uploaded_csv.seek(0)
        train_df = pd.read_csv(uploaded_csv)
    except Exception as e:
        csv_load_error = f"Error loading uploaded CSV: {e}"
elif use_defaults and os.path.exists(DEFAULT_CSV_FILE):
    try:
        train_df = load_csv_from_path(DEFAULT_CSV_FILE)
    except Exception as e:
        csv_load_error = f"Error loading CSV file '{DEFAULT_CSV_FILE}': {e}"
else:
    csv_load_error = "No CSV provided. Upload a CSV in the sidebar or place the default file in the app folder."

# ---------- Show status ----------
col_status_1, col_status_2 = st.columns(2)
with col_status_1:
    if model is not None:
        st.success("✅ Model loaded")
        st.caption(f"Model type: `{type(model).__name__}`")
    else:
        st.error(model_load_error)

with col_status_2:
    if train_df is not None:
        st.success(f"✅ CSV loaded ({len(train_df):,} rows)")
        st.caption(f"Columns: {', '.join(list(train_df.columns[:10]))}{'...' if len(train_df.columns)>10 else ''}")
    else:
        st.error(csv_load_error)

# Stop if model missing (can't predict)
if model is None:
    st.info("Provide a model to enable prediction.")
    st.stop()

# ---------- Build LabelEncoder for department (if possible) ----------
le = None
dept_options = []
if train_df is not None and "department" in train_df.columns:
    try:
        le = LabelEncoder()
        le.fit(train_df["department"].astype(str).unique().tolist())
        dept_options = list(le.classes_)
    except Exception:
        le = None
        dept_options = []

# ---------- Dashboard layout ----------
left, right = st.columns((2, 3))

with left:
    st.markdown("### Input Features")
    with st.form("predict_form"):
        # sensible defaults from training CSV if available
        def safe_median(col, fallback):
            try:
                if train_df is not None and col in train_df.columns:
                    return float(train_df[col].median())
            except Exception:
                pass
            return fallback

        age = st.number_input("Age", min_value=0, max_value=200, value=int(safe_median("age", 25)))
        experience = st.number_input("Experience (years)", min_value=0.0, max_value=100.0,
                                     value=float(safe_median("experience", 1.0)), step=0.5)
        salary = st.number_input("Salary", min_value=0.0, value=float(safe_median("salary", 30000.0)), step=100.0, format="%.2f")
        if dept_options:
            department = st.selectbox("Department", options=dept_options)
        else:
            department = st.text_input("Department (type)")

        st.write("")  # spacing
        predict_btn = st.form_submit_button("🔮 Predict Score")

    # Extra: let user save encoder mapping (optional)
    if le is not None:
        if st.checkbox("Show department encoding mapping"):
            mapping_df = pd.DataFrame({"department": list(le.classes_), "encoded": le.transform(list(le.classes_))})
            st.table(mapping_df)

with right:
    st.markdown("### Prediction & Insights")
    # Placeholder containers
    pred_placeholder = st.empty()
    info_placeholder = st.empty()
    viz_placeholder = st.container()

# ---------- Prediction logic ----------
if predict_btn:
    # Validate inputs
    if any(v is None or (isinstance(v, str) and v.strip() == "") for v in [age, experience, salary, department]):
        st.error("All features are required. Please fill every field.")
    else:
        # encode department
        if le is not None:
            dept_str = str(department)
            if dept_str in le.classes_:
                dept_enc = int(le.transform([dept_str])[0])
            else:
                # unseen -> sentinel
