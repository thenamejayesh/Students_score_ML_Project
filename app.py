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


def infer_n_features(estimator) -> Optional[int]:
    try:
        n = getattr(estimator, "n_features_in_", None)
        if n is not None:
            return int(n)
    except Exception:
        pass
    # fallback: coef_ shape
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


# ----------------- Load defaults (if present) -----------------

def load_default_csv(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def human_dtype(col: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(col):
        return "numeric"
    else:
        return "categorical"


# ----------------- Sidebar: Uploads & Settings -----------------
# Sidebar removed as requested
show_raw = False
enable_batch = False
uploaded_model = None
uploaded_csv = None("Enable batch prediction (CSV -> predictions)", value=True)

# try loading dataset
user_df = None
if uploaded_csv is not None:
    try:
        user_df = pd.read_csv(uploaded_csv)
        st.sidebar.success("Uploaded CSV loaded")
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")
else:
    user_df = load_default_csv(DEFAULT_CSV)
    if user_df is not None:
        st.sidebar.info(f"Loaded default CSV: {DEFAULT_CSV}")

# try loading model
model_obj = None
if uploaded_model is not None:
    try:
        # Uploaded file is a BytesIO-like object; seek to start
        uploaded_model.seek(0)
        model_obj = safe_load_pickle_file(uploaded_model)
        st.sidebar.success("Uploaded model loaded")
    except Exception as e:
        st.sidebar.error(str(e))
else:
    if os.path.exists(DEFAULT_MODEL):
        try:
            model_obj = load_model_from_path(DEFAULT_MODEL)
            st.sidebar.info(f"Loaded default model: {DEFAULT_MODEL}")
        except Exception as e:
            st.sidebar.warning(f"Could not load default model: {e}")
    else:
        st.sidebar.warning("No default model found. Upload a .pkl in the sidebar.")

# extract estimator if needed
estimator = extract_estimator(model_obj) if model_obj is not None else None

# ----------------- Main layout -----------------
col1, col2 = st.columns((2, 1))

with col1:
    st.title("Score Predictor — Interactive Dashboard")
    st.markdown(
        "Use the sidebar to upload a model (pickle) and/or a CSV.\n"
        "The app will try to create dynamic inputs from the dataset columns so you can make single predictions or batch predictions."
    )

    if user_df is None:
        st.info("No dataset available. Upload a CSV in the sidebar or place one at `/mnt/data/student_scores (1).csv`.")
    else:
        if show_raw:
            st.subheader("Dataset preview")
            st.dataframe(user_df.head(10))

    st.markdown("---")

    # If dataset exists, create dynamic input form
    if user_df is not None:
        st.subheader("Create input (single prediction)")
        # Use first row as defaults where possible
        sample = user_df.head(1).copy()
        # Ask user which column is target (optional)
        target_col = st.selectbox("(Optional) Select target/label column (will be ignored for inputs)", options=[None] + user_df.columns.tolist())

        # Build list of feature columns
        feature_cols = [c for c in user_df.columns.tolist() if c != target_col]
        st.info(f"Detected feature columns: {feature_cols}")

        # Make a form for input
        with st.form("single_input_form"):
            input_data = {}
            for col in feature_cols:
                col_ser = user_df[col]
                dtype = human_dtype(col_ser)
                if dtype == "numeric":
                    mn = float(np.nanmin(col_ser)) if not col_ser.isna().all() else 0.0
                    mx = float(np.nanmax(col_ser)) if not col_ser.isna().all() else 1.0
                    mean = float(np.nanmean(col_ser)) if not col_ser.isna().all() else 0.0
                    # give reasonable slider/number input
                    if (mx - mn) > 1000 or mx == mn:
                        val = st.number_input(col, value=mean)
                    else:
                        step = (mx - mn) / 100 if (mx - mn) != 0 else 1.0
                        val = st.slider(col, min_value=mn, max_value=mx, value=mean, step=step)
                    input_data[col] = val
                else:
                    # treat as categorical: provide selectbox with top values
                    uniques = col_ser.dropna().unique().tolist()
                    default = uniques[0] if uniques else ""
                    val = st.selectbox(col, options=uniques + ["Other (type below)"], index=0 if uniques else 0)
                    if val == "Other (type below)":
                        val = st.text_input(f"Enter value for {col}", value="")
                    input_data[col] = val

            submitted = st.form_submit_button("Predict single")

        if submitted:
            if estimator is None:
                st.error("No model loaded. Upload a .pkl model in the sidebar or place one at the default path.")
            else:
                # prepare dataframe for prediction
                X = pd.DataFrame([input_data])
                # Auto-drop extra column if model expects 3 features
                try:
                    if infer_n_features(estimator) == 3 and X.shape[1] == 4:
                        X = X.iloc[:, :-1]
                except Exception:
                    pass
                st.write("Input to be sent to model:")
                st.dataframe(X)
                try:
                    # If saved model is a pipeline that expects exact columns, try to send full X
                    pred = estimator.predict(X)
                except Exception:
                    # fallback: send numpy array
                    try:
                        pred = estimator.predict(X.values)
                    except Exception as e:
                        st.error("Prediction failed. See exception below.")
                        st.exception(e)
                        pred = None

                if pred is not None:
                    val = pred[0] if hasattr(pred, "__len__") else pred
                    st.success(f"Predicted value: {val}")
                    # show additional info
                    if hasattr(estimator, "predict_proba"):
                        try:
                            proba = estimator.predict_proba(X)
                            st.write("Model predict_proba output:")
                            st.write(proba)
                        except Exception:
                            pass

    else:
        st.info("Upload a CSV to create a dynamic input form.")

    st.markdown("---")

    # Batch prediction (CSV)
    if enable_batch and user_df is not None and estimator is not None:
        st.subheader("Batch predict: run predictions on entire dataset")
        if st.button("Run batch prediction on dataset"):
            try:
                X_batch = user_df.copy()
                # Auto-drop extra column if model expects 3 features
                try:
                    if infer_n_features(estimator) == 3 and X_batch.shape[1] == 4:
                        if target_col is not None and target_col in X_batch.columns:
                            X_batch = X_batch.drop(columns=[target_col])
                        else:
                            X_batch = X_batch.iloc[:, :-1]
                except Exception:
                    pass
                if target_col is not None and target_col in X_batch.columns:
                    X_batch = X_batch.drop(columns=[target_col])
                try:
                    preds = estimator.predict(X_batch)
                except Exception:
                    preds = estimator.predict(X_batch.values)

                # append predictions
                out = user_df.copy()
                out["_prediction"] = preds
                st.success("Batch prediction finished — preview below")
                st.dataframe(out.head(20))
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")
            except Exception as e:
                st.error("Batch prediction failed")
                st.exception(e)

# Right column removed completely as sidebar removed
# End of file
