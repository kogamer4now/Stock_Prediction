import os, sys, warnings, json
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
from sagemaker.serializers import CSVSerializer
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Access the secrets
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

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

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_pca.shap',
    "pipeline":  'finalized_pca_model.tar.gz',
    "keys":   ["IBM"],
    "inputs": [{"name": k, "type": "number", "min": 0.0, "default": 100.0, "step": 10.0} for k in ["IBM"]]
}

TAR_LOCAL_PATH = os.path.join(tempfile.gettempdir(), 'finalized_pca_model.tar.gz')


def download_tar_if_needed(_session, bucket):
    """Downloads the model tar.gz from S3 into temp dir if not already present."""
    if not os.path.exists(TAR_LOCAL_PATH):
        s3_client = _session.client('s3')
        s3_client.download_file(
            Bucket=bucket,
            Key=f"sklearn-pipeline-deployment/{MODEL_INFO['pipeline']}",
            Filename=TAR_LOCAL_PATH
        )


def load_pipeline(_session, bucket):
    """Downloads the tar.gz and loads the joblib pipeline from it."""
    download_tar_if_needed(_session, bucket)
    with tarfile.open(TAR_LOCAL_PATH, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_file))


def load_csv_from_tar(_session, bucket):
    """Extracts SP500Data.csv from inside the model tar.gz."""
    download_tar_if_needed(_session, bucket)
    csv_local_path = os.path.join(tempfile.gettempdir(), 'SP500Data.csv')
    if not os.path.exists(csv_local_path):
        with tarfile.open(TAR_LOCAL_PATH, "r:gz") as tar:
            csv_members = [m for m in tar.getnames() if m.endswith('SP500Data.csv')]
            if not csv_members:
                raise FileNotFoundError("SP500Data.csv not found inside the model tar.gz")
            tar.extract(csv_members[0], path=tempfile.gettempdir())
            extracted_path = os.path.join(tempfile.gettempdir(), csv_members[0])
            if extracted_path != csv_local_path:
                os.replace(extracted_path, csv_local_path)
    return pd.read_csv(csv_local_path, index_col=0)


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


def call_model_api(user_inputs):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    try:
        raw_pred = predictor.predict(user_inputs)
        pred_val = float(np.array(raw_pred).ravel()[-1])
        return round(pred_val, 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


def display_explanation(user_inputs, session, aws_bucket):
    explainer_name       = MODEL_INFO["explainer"]
    local_explainer_path = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        local_explainer_path
    )

    # SP500Data.csv is bundled inside the tar.gz — extract it from there
    dataset = load_csv_from_tar(session, aws_bucket)

    ibm_price    = float(user_inputs['IBM'])
    closest_date = (dataset['IBM'] - ibm_price).abs().idxmin()

    return_period = 5
    X = np.log(dataset.drop(['NOC'], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + "_CR_Cum" for name in X.columns]
    input_row = X.loc[[closest_date]]

    best_pipeline          = load_pipeline(session, aws_bucket)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_transformed      = preprocessing_pipeline.transform(input_row)

    n_comps       = input_transformed.shape[1]
    feature_names = [f'KernelPCA_{i+1}' for i in range(n_comps)]
    input_df      = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], value=inp['default'], step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    res, status = call_model_api(user_inputs)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(user_inputs, session, aws_bucket)
    else:
        st.error(res)
