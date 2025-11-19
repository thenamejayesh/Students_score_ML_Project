 # app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Config / default paths
# -----------------------
DEFAULT_MODEL_PATH = "/mnt/data/Student_model (3).pkl"
DEFAULT_DATA_PATH = "/mnt/data/student_scores (1).csv"
st.set_page_config(page_title="Dynamic Feature Predictor", layout="centered")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_pickle(path_or_file):
    try:
        if hasattr(path_or_file, "read"):
            # file-like from uploader
            return pickle.load(path_or_file)
        else:
            with open(path_or_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        raise

def is_numeric_series(s: pd.Series):
    return pd.api.types.is_numeric_dtype(s)

def fit_label_encoders(df, cat_cols):
    encs = {}
    for c in cat_cols:
        le = LabelEncoder()
        # convert to str to handle mixed types
        encs[c] = le.fit(df[c].astype(str))
    return encs

def transform_with_encoders(df_row, encoders):
    out = df_row.copy()
    for c, le in encoders.items():
        try:
            out[c] = le.transform(pd.Series([str(out[c])]))[0]
        except Exception:
            # if value not seen during fit, fit-transform fallback (not ideal but robust)
            # create a small fallback encoder to encode that single value as 0/1 scheme
            tmp = LabelEncoder()
            tmp.fit([str(out[c])])
            out[c] = tmp.transform([str(out[c])])[0]
    return out

def infer_input_features(df):
    """
    Heuristic: If CSV has an obvious target column (named 'target' or 'score' or 'y' or 'label' or 'marks'),
    remove it from inputs. Otherwise use all columns as input features.
    """
    if df is None:
        return []
    candidates = [c.lower() for c in df.columns]
    target_names = {"target", "score", "marks", "marks_obtained", "y", "label", "result"}
    found = None
    for t in target_names:
        if t in candidates:
            found = df.columns[candidates.index(t)]
            break
    if found:
        input_feats = [c for c in df.columns if c != found]
        return input_feats, found
    else:
        return list(df.columns), None

def model_expected_feature_count(est):
    n = None
    try:
        n = int(getattr(est, "n_features_in_", None)) if getattr(est, "n_features_in_", None) is not None else None
    except Exception:
        n = None
    if n is None:
        try:
            coef = getattr(est, "coef_", None)
            if coef is not None:
                arr = np.array(coef)
                if arr.ndim == 1:
                    n = arr.shape[0]
                elif arr.ndim == 2:
                    n = arr.shape[1]
        except Exception:
            n = None
    return n

def model_feature_names(est):
    try:
        return list(getattr(est, "feature_names_in_", [])) or None
    except Exception:
        return None

# -----------------------
# Load defaults (cached)
# -----------------------
default_df = load_csv(DEFAULT_DATA_PATH)
# -----------------------
# Sidebar: upload files
# -----------------------
st.sidebar.title("Files & Settings")
uploaded_csv = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"])

# Prefer uploaded CSV, else default
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")
        df = default_df
else:
    df = default_df
    if df is not None:
        st.sidebar.info(f"Using default CSV at `{DEFAULT_DATA_PATH}`")
    else:
        st.sidebar.warning("No CSV found. Upload one to enable dataset-driven features.")

# Load model if provided or default
model = None
if uploaded_model is not None:
    try:
        model = load_pickle(uploaded_model)
        st.sidebar.success("Uploaded model loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not load uploaded model: {e}")
else:
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            model = load_pickle(DEFAULT_MODEL_PATH)
            st.sidebar.info(f"Loaded model from `{DEFAULT_MODEL_PATH}`")
        except Exception as e:
            st.sidebar.warning("Default model exists but could not be loaded.")

# If model is dict-like with encoder
saved_encoders = None
if isinstance(model, dict):
    # Accept dicts like {'model': estimator, 'encoders': {...}} or {'model': estimator, 'encoder': le}
    if "encoders" in model and isinstance(model["encoders"], dict):
        saved_encoders = model["encoders"]
    elif "encoder" in model:
        # single encoder for one categorical column — not common, but keep for compatibility
        saved_encoders = {"department": model["encoder"]} if model.get("encoder") is not None else None

# -----------------------
# Infer features from CSV
# -----------------------
if df is not None:
    input_features, detected_target = infer_input_features(df)
    st.sidebar.write(f"Inferred input features from CSV: {input_features}")
    if detected_target:
        st.sidebar.write(f"Detected target column (excluded from inputs): `{detected_target}`")
else:
    input_features = []
    detected_target = None

# Decide for each feature whether numeric or categorical (if dataset exists)
feature_types = {}
cat_values = {}
if df is not None and input_features:
    for f in input_features:
        if is_numeric_series(df[f]):
            feature_types[f] = "numeric"
            cat_values[f] = None
        else:
            feature_types[f] = "categorical"
            # unique options for selectbox
            cat_values[f] = sorted(df[f].astype(str).dropna().unique().tolist())

# Fit label encoders for categorical features (from dataset), unless saved_encoders exist
encoders = {}
for f, t in feature_types.items():
    if t == "categorical":
        if saved_encoders and f in saved_encoders:
            encoders[f] = saved_encoders[f]
        else:
            le = LabelEncoder()
            # if column exists in df, fit on its values; else fit on available cat_values
            if df is not None and f in df.columns:
                le.fit(df[f].astype(str).fillna("##MISSING##"))
            else:
                le.fit([str(x) for x in (cat_values.get(f) or ["unknown"])])
            encoders[f] = le

# -----------------------
# Build dynamic UI for inputs
# -----------------------
st.title("Predictor (uses dataset features)")

if not input_features:
    st.info("No dataset available to infer features. Upload a CSV or place it at the default path.")
    # Still allow user to upload a model and attempt a manual predict (not implemented further)
else:
    st.write("Enter values for the following features (derived from your dataset):")
    user_inputs = {}
    # Use columns order as inferred
    for f in input_features:
        t = feature_types.get(f, "numeric")
        if t == "numeric":
            # infer sensible min/max/default from dataset if available
            if df is not None and f in df.columns and is_numeric_series(df[f]):
                col = df[f].dropna()
                default = float(col.median()) if len(col) > 0 else 0.0
                minv = float(col.min()) if len(col) > 0 else 0.0
                maxv = float(col.max()) if len(col) > 0 else default if default != 0 else 100.0
                # provide a number_input with reasonable step
                step = (maxv - minv) / 100.0 if maxv > minv else 1.0
                user_inputs[f] = st.number_input(f"{f} (numeric)", value=float(default), min_value=minv, max_value=maxv, step=step, format="%.4f")
            else:
                user_inputs[f] = st.number_input(f"{f} (numeric)", value=0.0, format="%.4f")
        else:
            options = cat_values.get(f) or ["option1"]
            user_inputs[f] = st.selectbox(f"{f} (categorical)", options=options)

    st.markdown("---")

    # Show selected values
    st.subheader("Selected input values")
    st.write(user_inputs)

    # Show model info
    if model is None:
        st.warning("No model loaded. Upload a .pkl model in the sidebar to enable prediction.")
    else:
        # unwrap model if dict
        if isinstance(model, dict) and "model" in model:
            estimator = model["model"]
        else:
            estimator = model

        expected_n = model_expected_feature_count(estimator)
        feat_names_from_model = model_feature_names(estimator)

        st.write("Model diagnostics:")
        st.write(f"- Model expects {expected_n if expected_n is not None else 'unknown'} features.")
        if feat_names_from_model:
            st.write(f"- Model.feature_names_in_: {feat_names_from_model}")

        # Prepare row
        row = {}
        for f in input_features:
            row[f] = user_inputs[f]

        # If categorical, transform using fitted encoders
        for f, t in feature_types.items():
            if t == "categorical":
                if f in encoders and encoders[f] is not None:
                    try:
                        row[f] = int(encoders[f].transform([str(row[f])])[0])
                    except Exception:
                        # fallback: fit small encoder for this unseen value
                        tmp = LabelEncoder()
                        tmp.fit([str(row[f])])
                        row[f] = int(tmp.transform([str(row[f])])[0])
                else:
                    # leave as string; model may fail if it expects numeric
                    row[f] = str(row[f])

        # Build DataFrame in dataset order
        X_df = pd.DataFrame([row], columns=input_features)

        st.write("Prepared input DataFrame sent to model (after encoding):")
        st.dataframe(X_df)

        # Align to model.feature_names_in_ if available
        if feat_names_from_model is not None:
            # If model expects a subset/order, try to select/reorder
            missing = [c for c in feat_names_from_model if c not in X_df.columns]
            if missing:
                st.warning(f"Model expects columns not present in dataset-driven features: {missing}")
                # allow user to select columns to drop from X_df or provide defaults (not ideal)
            else:
                X_to_feed = X_df[feat_names_from_model].values
        else:
            X_to_feed = X_df.values

        # If feature count mismatch, provide options
        if expected_n is not None and X_to_feed.shape[1] != expected_n:
            st.warning(f"Feature count mismatch: prepared input has {X_to_feed.shape[1]} features but model expects {expected_n}.")
            st.write("Options:")
            st.write(" - If model was trained without some dataset columns (for example `department` was preprocessed/removed), select which columns should be sent to the model in the order it expects.")
            chosen = st.multiselect("Select & order columns to send to model (top -> leftmost)", options=list(X_df.columns), default=list(X_df.columns))
            if len(chosen) == 0:
                st.error("No columns selected to feed the model.")
                X_to_feed = None
            else:
                X_to_feed = X_df[chosen].values

        # Final predict
        if st.button("Predict"):
            if X_to_feed is None:
                st.error("No valid input matrix to feed model.")
            else:
                try:
                    # some saved models are dicts -> access 'model' key above already
                    pred = estimator.predict(X_to_feed)
                    pred_val = pred[0] if hasattr(pred, "__len__") else pred
                    st.success(f"Prediction result: {pred_val}")
                    st.write("Final matrix shape sent to model:", X_to_feed.shape)
                except Exception as e:
                    st.error("Prediction failed. See exception below.")
                    st.exception(e)

st.markdown("---")
st.write("Notes:")
st.write("- This app prefers the dataset's features. If your model was trained on a different set or order of features, use a training-time pipeline and save it together with the model.")
st.write("- Best practice: Save a single object like `{'model': pipeline, 'encoders': {'colA': leA, ...}}` so the app can reuse the exact preprocessing.")
