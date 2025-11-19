import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# --- CONFIG: update these paths if your files are named differently ---
MODEL_PATH = "/mnt/data/Student_model (1).pkl"
DATA_CSV_PATH = "/mnt/data/student_scores (1).csv"
# -----------------------------------------------------------------------

st.set_page_config(page_title="Streamlit Prediction App", layout="centered")

st.title("Prediction App — Enter all features and press Predict")
st.markdown(
    "This app loads a pickled model and the training CSV to build a label-encoder for categorical columns (department). "
    "All inputs are required before prediction."
)

@st.cache_data(show_spinner=False)
def load_model(path):
    if not o
