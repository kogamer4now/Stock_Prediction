import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import shap

from joblib import load

# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(current_dir, 'X_train.csv')

dataset = pd.read_csv(file_path)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Access the secrets
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "shap_explainer.joblib",
    "keys"      : ['int_rate', 'dti', 'revol_util', 'annual_inc', 'loan_amnt', 'term', 'emp_length'],
    "inputs"    : [
        {"name": "int_rate",   "type": "number", "min": 5.0,    "max": 30.0,    "default": 13.5,   "step": 0.1},
        {"name": "dti",        "type": "number", "min": 0.0,    "max": 50.0,    "default": 15.0,   "step": 0.5},
        {"name": "revol_util", "type": "number", "min": 0.0,    "max": 100.0,   "default": 50.0,   "step": 1.0},
        {"name": "annual_inc", "type": "number", "min": 10000.0,"max": 500000.0,"default": 75000.0,"step": 1000.0},
        {"name": "loan_amnt",  "type": "number", "min": 1000.0, "max": 40000.0, "default": 15000.0,"step": 500.0},
        {"name": "term",       "type": "number", "min": 36.0,   "max": 60.0,    "default": 36.0,   "step": 24.0},
        {"name": "emp_length", "type": "number", "min": 0.0,    "max": 10.0,    "default": 5.0,    "step": 1.0},
    ]
}


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)

    with open(local_path, "rb") as f:
        return load(f)


# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = raw_pred['prediction'][0]
        mapping  = {0: "Fully Paid", 1: "Default"}
        return mapping.get(int(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        'loan-default-model/shap_explainer.joblib',
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    expected_features = list(dataset.columns)
    input_row = pd.DataFrame([input_df])[expected_features].astype(float)

    shap_values = explainer.shap_values(input_row)

    st.subheader("Decision Transparency (SHAP)")

    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_row.iloc[0].values,
        feature_names=expected_features
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    top_feature = pd.Series(shap_values[0], index=expected_features).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Risk Prediction")

with st.form("pred_form"):
    st.subheader("Borrower & Loan Inputs")
    cols        = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

expected_features = list(dataset.columns)
original = {k: float(v) for k, v in dataset.iloc[0:1].to_dict(orient='records')[0].items()}

for k, v in user_inputs.items():
    if k in original:
        original[k] = float(v)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
