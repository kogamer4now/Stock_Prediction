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
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump
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
    "pipeline"  : "loan_default_model.tar.gz",
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


def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(f"{joblib_file}")


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client  = _session.client('s3')
    local_path = local_path

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)


# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping  = {0: "Fully Paid", 1: "Default"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer      = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline         = load_pipeline(session, aws_bucket, 'loan-default-model')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df              = pd.DataFrame(input_df)
    input_df_transformed  = preprocessing_pipeline.transform(input_df)

    dataset_1      = dataset.iloc[:, 0:]
    feature_names  = dataset_1.columns[1:]
    selector       = best_pipeline.named_steps['selector']
    selected_features     = feature_names[selector.get_support()]
    input_df_transformed  = pd.DataFrame(input_df_transformed, columns=selected_features)

    shap_values = explainer(input_df_transformed)

    st.subheader("Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])  # class 1 = default
    st.pyplot(fig)
    top_feature = pd.Series(
        shap_values[0, :, 1].values,
        index=shap_values[0, :, 1].feature_names
    ).abs().idxmax()
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

# Get the exact 25 features the model was trained on
expected_features = list(dataset.columns)
original = {k: float(v) for k, v in dataset.iloc[0:1].to_dict(orient='records')[0].items()}

# Update with user inputs but only for features that exist in training data
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
