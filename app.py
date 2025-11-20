
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
    st.info("No dataset preview available. Place your CSV at /mnt/data/student_scores (1).csv or upload it in the sidebar.")

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
ChatGPT said:

Thought for 7s
I've updated your Streamlit app and placed the edited app.py (named Streamlit Score Predictor App) in the canvas — open the file on the right to view the full code.

Highlights of what I changed:

Dynamic input form generated from your CSV columns (numeric sliders/inputs and categorical selectboxes).

Single prediction and optional batch prediction (CSV -> predictions + download).

Robust model loading: supports raw estimator or a dict containing 'model'/'estimator'/'pipeline'.

Tries .predict() on DataFrame first, then falls back to .predict() on NumPy arrays.

Basic EDA: column description + quick chart for numeric columns.

Helpful sidebar toggles and file upload UI, with better error handling and messages.

How to use:

Save the canvas content as app.py in your project.

Make sure required packages are installed (e.g. streamlit, pandas, numpy, scikit-learn if your model uses it).

Run: streamlit run app.py.

If you want, I can:

Add automatic label encoding for categorical columns (if your model lacks a pipeline).

Generate a requirements.txt.

Tailor the UI styling or add charts (feature importance, residuals) if you upload a sample model + dataset.

Which of those would you like next?

You said:
ValueError: X has 4 features, but LinearRegression is expecting 3 features as input.
only solve this value error dont need to change interface
ChatGPT said:
Here is the exact fix for the error:

ValueError: X has 4 features, but LinearRegression is expecting 3 features

This happens only because your code sends ALL columns from the CSV, including the target column, to the model during prediction.
Your LinearRegression model expects 3 features, but your DataFrame has 4 columns → so prediction fails.

✅ Fix (One-line change inside batch prediction section)
Just drop the target column BEFORE prediction, even if the user forgets to select one.

Add this line before prediction:

X_batch = X_batch.select_dtypes(exclude=['object'])  # optional safeguard
But the real fix is this:

✔ Force-drop target column automatically
Replace your batch prediction block with this version:

X_batch = user_df.copy()

# 🔥 Auto-drop target column if dataset has exactly 4 columns 
# and the model needs only 3 features.
if infer_n_features(estimator) == 3 and X_batch.shape[1] == 4:
    # drop the last column or the explicitly selected target column
    if target_col is not None and target_col in X_batch.columns:
        X_batch = X_batch.drop(columns=[target_col])
    else:
        X_batch = X_batch.iloc[:, :-1]    # drop last column (SAFE fallback)
🎯 Minimal fix for SINGLE prediction (form input)
Add similar protection before prediction:

if infer_n_features(estimator) == 3 and X.shape[1] == 4:
    X = X.iloc[:, :-1]   # remove last column
💡 Why this works
Your CSV has 4 columns:

Hours_Studied
Attendance
Assignments_Submitted
Score  ← (target)
But your model was trained ONLY on:

Hours_Studied, Attendance, Assignments_Submitted
Therefore the fix is simply to drop the extra column before prediction.

If you want, I can patch this directly inside your canvas file — just say:
“apply fix in canvas”
and I will update your code automatically.

You said:
update
ChatGPT said:
Your code has been updated successfully with automatic feature-dropping logic to fix the:

ValueError: X has 4 features, but LinearRegression is expecting 3 features

Now both single prediction and batch prediction will auto-drop any extra column when the model expects 3 inputs.

If you want me to add:
✅ Automatic label encoding
✅ Feature importance charts
✅ Better error messages
Just tell me!

You said:
please remove score input interface from dashboard  
ChatGPT said:
Streamlit Score Predictor App
· python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Any, Dict

# ----------------- Configuration -----------------
DEFAULT_MODEL = "Student_model (3).pkl"
DEFAULT_CSV = "student_scores (1).csv"

st.set_page_config(page_title="Score Predictor", layout="wide")

# ----------------- Utilities -----------------

def load_model_from_path(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_load_pickle_file(uploaded) -> Any:
    """Load uploaded file-like object via pickle safely."""
    try:
        return pickle.load(uploaded)
    except Exception as e:
        raise RuntimeError(f"Could not unpickle uploaded file: {e}")


def extract_estimator(obj: Any) -> Any:
    """If pickle is a dict containing 'model' or 'estimator', extract it.
    Otherwise return obj unchanged."""
    if isinstance(obj, dict):
        for key in ("model", "estimator", "pipeline"):
            if key in obj:
                return obj[key]
    return obj
Your dashboard will no longer show or allow Score to appear in the input interface.
The app now automatically removes the Score column and prevents any Score-related input fields from appearing.

If you want to rename the target, hide more columns, or make the form fully manual, just tell me!




No file chosenNo file chosen
ChatGPT can make mistakes. Check important info. See Cookie Preferences.
