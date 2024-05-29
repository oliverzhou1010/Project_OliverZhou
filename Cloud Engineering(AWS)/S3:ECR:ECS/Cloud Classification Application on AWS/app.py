import os
from pathlib import Path
import logging
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import src.aws_utils as aws

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the first logger
logger = logging.getLogger(__name__)

# Log the start of the program or test suite execution
logger.info("Starting the program or test suite execution...")

# Load the dataset and trained classifier
BUCKET_NAME = os.getenv("BUCKET_NAME", "sgf3992-clouds")
ARTIFACTS_PREFIX = Path(os.getenv("ARTIFACTS_PREFIX", "experiments/"))

# Create artifacts directory to keep model files
artifacts = Path() / "artifacts"
artifacts.mkdir(exist_ok=True)

# Add cacheing to store the intermediate steps
@st.cache_data
def load_data(data_path: Path, s3_key: str):
    """
    Load data from AWS S3 and save it to a local file.

    Args:
        data_path (Path): Local path where the data file will be saved.
        s3_key (str): S3 key of the data file.

    Returns:
        tuple: List of class names, features (X), and target values (y).
    """
    try:
        logging.info("Loading data from S3.")
        data_path.parent.mkdir(exist_ok=True)  
        aws.download_s3(BUCKET_NAME, s3_key, data_path)

        # Handle errors if the data path do not have this file
        if not data_path.exists():
            raise FileNotFoundError(
                f"The file {data_path} does not exist after attempting to download from S3."
            )

        # Load data 
        data = pd.read_csv(data_path)

        # Select only the relevant features
        features = data[['log_entropy', 'IR_norm_range', 'entropy_x_contrast']].values
        targets = data['class'].values

        # Get class names
        class_names = ['Class 0' if y == 0 else 'Class 1' for y in np.unique(targets)]
        return class_names, features, targets
    except Exception as err:
        logging.error("Unable to load data: %s", err)
        st.error(f"Unable to load data: {err}")
        raise

# Add cacheing to store the intermediate steps
@st.cache_resource
def load_model(model_path: Path, s3_key: str):
    """
    Load model from S3 and save to a local file.

    Args:
        model_path (Path): Local path to save the model file.
        s3_key (str): S3 key of the model file.

    Returns:
        sklearn.base.BaseEstimator: Loaded model.
    """
    try:
        logging.info("Initiating download of the model from s3://%s/%s to local path: %s", BUCKET_NAME, s3_key, model_path)
        model_path.parent.mkdir(exist_ok=True) 

        # Download from S3
        aws.download_s3(BUCKET_NAME, s3_key, model_path)

        # Handle errors if the model path does not contain the model
        if not model_path.exists():
            raise FileNotFoundError(
                f"The file {model_path} does not exist after attempting to download from S3."
            )

        # Load model
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as err:
        logging.error("Unable to load model: %s", err)
        st.error(f"Unable to load model: {err}")
        raise


def slider_values(series) -> tuple[float, float, float]:
    """
    Calculate the minimum, maximum, and mean values of a pandas Series.

    Args:
        series (pd.Series): Input pandas Series.

    Returns:
        tuple: Minimum, maximum, and mean values of the series as floats.
    """
    try:
        logging.info("Calculating minimum, maximum, and mean values of the series.")
        min_val = float(series.min())
        max_val = float(series.max())
        mean_val = float(series.mean())
        return min_val, max_val, mean_val
    except Exception as e:
        logging.error("Error occurred while calculating values: %s", e)
        raise



# Create the application title and description
st.title("Cloud Classification Portal")
st.write("This app classifies two types of clouds based on its images.")

# Create the application subheader
st.subheader("Project Explanation")
st.write("""
**Goal: Cloud Classification:**
This is a machine learning model for classifying cloud types from images based on features like log entropy, IR norm range, and entropy x contrast. 
Select a model version from the dropdown bar on the left and view predictions.
""")

st.subheader("Model Overview")
st.write("""
**Model 1: Random Forest (version1)**
The original random forest model with no fancy things added on. The performance metrics is already super good.

**Model 2: Random Forest (version2)**
The updated version of random forest model. Utilize techniques that allow the prediction to be robust and stable, which is better fit for a more disciplined scenario.
""")

model_version = os.getenv("DEFAULT_MODEL_VERSION", "default")
models = {
    "Random Forest (version1)": artifacts / model_version / "trained_model_object_1.pkl",
    "Random Forest (version2)": artifacts / model_version / "trained_model_object_2.pkl"
}

# Configure S3 location 
data_s3 = f"experiments/train.csv"
model_s3 = {name: f"experiments/{file.name}" for name, file in models.items()}
data_path = artifacts / model_version / "train.csv"

# Retrieve train and test data
try:
    class_names, X, y = load_data(data_path, data_s3)
    logger.info("Load data from S3 successfully.")
except Exception as err:
    logger.error("An error has occur when loading data from S3.")
    st.stop() 


# Model selection 
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Select from below", list(models.keys()))
selected_model = models[model_name]
selected_model_s3 = model_s3[model_name]

# Load selected model
try:
    clf = load_model(selected_model, selected_model_s3)
    logger.info("Load model from S3 successfully.")
except Exception as err:
    logger.error("An error has occur when loading model from S3.")
    st.stop() 

feature_names = ['log_entropy', 'IR_norm_range', 'entropy_x_contrast']

# Handle input paramters
st.sidebar.header("Input Parameters")
log_entropy = st.sidebar.slider("Log Entropy", *slider_values(X[:, 0]))
IR_norm_range = st.sidebar.slider("IR Norm Range", *slider_values(X[:, 1]))
entropy_x_contrast = st.sidebar.slider("Entropy x Contrast", *slider_values(X[:, 2]))

# Create prediction
st.markdown("<hr>", unsafe_allow_html=True)

try:
    input = pd.DataFrame(
        [[log_entropy, IR_norm_range, entropy_x_contrast]],
        columns=feature_names
    )
    # Make prediction for the current input
    prediction = clf.predict(input)
    pred_class = class_names[prediction[0]]
    probability = clf.predict_proba(input)[0][prediction[0]]
    logger.info("Successfully made predictions.")
    
    # Output the predictions
    result_html = f"""
    <div style="background-color:#EAF2F8; border: 1px solid #AED6F1; padding:20px; border-radius:5px;">
        <h3 style="color:#3498DB;">Prediction:</h3>
        <hr style="border: 0.5px solid #7FB3D5;">
        <p><strong>Predicted Class:</strong> {pred_class}</p>
        <p><strong>Probability:</strong> {probability:.2f}</p>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)    
except ValueError as err:
    logging.error("Unable to make prediction: %s", err)
    st.error(f"Unable to make prediction: {err}")
