import streamlit as st
import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.base import BaseEstimator
import boto3
from botocore.exceptions import NoCredentialsError

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Function to download the model from S3
def download_model_from_s3(bucket_name, model_key, download_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, model_key, download_path)
        logging.info(f"Model downloaded from S3: {bucket_name}/{model_key}")
        return download_path
    except NoCredentialsError:
        logging.error("Credentials not available")
        return None
    except Exception as e:
        logging.error(f"Error downloading model from S3: {str(e)}")
        return None

# Load the model
def load_prediction_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        if not isinstance(model, BaseEstimator):
            logging.error("Loaded model is not a scikit-learn estimator.")
            return None
        return model
    except Exception as e:
        logging.error(f"Error loading prediction model: {str(e)}")
        return None

# Predict the species for input data
def predict(input_data, model):
    try:
        predictions = model.predict([input_data])
        predict_prob = model.predict_proba([input_data])
        return predictions,predict_prob
    
    except Exception as e:
        logging.error(f"Error predicting: {str(e)}")
        return None

# Get model input features
def get_model_input_features(model):
    try:
        # Assuming the model has been trained with a DataFrame
        return model.feature_names_in_
    except AttributeError:
        logging.error("The model does not have feature_names_in_ attribute.")
        return None

def run_app():
    st.title('Cancer Diagnosis Prediction App')
    st.header('Predict whether a tumor is Malignant (M) or Benign (B)')

    # # S3 configuration
    # bucket_name = 'vxi3373-mlds423-project-model'
    # model_keys = {
    #     'Model 0': 'model/model_a/model_A.pkl',  # Update paths as necessary
    #     'Model 1': 'model/model_b/model_B.pkl'
    # }
    
    # model_name = st.sidebar.selectbox('Select Model', list(model_keys.keys()))
    # model_key = model_keys[model_name]
    # model_path = f"/tmp/{model_key.split('/')[-1]}"  # Temporary local path

    # if download_model_from_s3(bucket_name, model_key, model_path):
    #     model = load_prediction_model(model_path)
    #     if model is not None:
    #         feature_names = get_model_input_features(model)
    #         if feature_names is not None:
    #             st.subheader(f'Input Features for {model_name}')
    #             input_data = []

    #             # Create input fields based on model's expected input features
    #             for feature in feature_names:
    #                 value = st.number_input(f'{feature}', min_value=-50.0, max_value=100.0, step=0.1)
    #                 input_data.append(value)
                
    #             # Predict
    #             predicted, predict_p = predict(input_data, model)
    #             if predicted and predict_p is not None:
    #                 logging.info(f'Predicted distribution: {predicted}')
                    
    #                 # Display the predicted result
    #                 st.subheader('Prediction:')
    #                 st.write(predicted)
                    
    #                 st.subheader('Predictio Confidence:')
    #                 st.write(predict_p[0])
    # S3 configuration
    bucket_name = 'vxi3373-mlds423-project-model'
    model_keys = {
        'Normal Randomforest': 'model/model_a/model_A.pkl',  # Update paths as necessary
        'Deep Randomforest': 'model/model_b/model_B.pkl'
    }
    
    st.sidebar.header('Model Selection')
    model_name = st.sidebar.selectbox('Select Model', list(model_keys.keys()))
    model_key = model_keys[model_name]
    model_path = f"/tmp/{model_key.split('/')[-1]}"  # Temporary local path

    if download_model_from_s3(bucket_name, model_key, model_path):
        model = load_prediction_model(model_path)
        if model is not None:
            feature_names = get_model_input_features(model)
            if feature_names is not None:
                st.sidebar.header('Input Features')
                st.sidebar.markdown('Please input the following features:')
                input_data = []

                # Create input fields based on model's expected input features
                for feature in feature_names:
                    value = st.sidebar.slider(f'{feature}', min_value=0.0, max_value=2500.0, step=0.1)
                    input_data.append(value)
                
                # Predict
                if st.sidebar.button('Predict'):
                    predicted, predict_p = predict(input_data, model)
                    if predicted and predict_p is not None:
                        logging.info(f'Predicted distribution: {predicted}')
                        # Display the predicted result
                        st.subheader('Prediction Result:')
                        result = 'Malignant (M)' if predicted[0] == 'M' else 'Benign (B)'
            
                        st.success(f'The tumor is predicted to be {result}')
                        
                        st.subheader('Prediction Confidence:')
                        st.success(f'The model makes prediction with this confidence {predict_p[:, 0][0]} (out of 1)')
                        
                        # Explanation
                        st.subheader('Explanation:')
                        if result == 'Malignant (M)':
                            st.write("""
                            **Malignant Tumor:**
                            - A malignant tumor is cancerous. This means that the cells are abnormal and can grow uncontrollably.
                            - Malignant tumors can invade nearby tissues and spread to other parts of the body through the blood and lymph systems.
                            - Early detection and treatment are crucial for a better prognosis.
                            """)
                        else:
                            st.write("""
                            **Benign Tumor:**
                            - A benign tumor is not cancerous. The cells in a benign tumor are normal and do not invade nearby tissues.
                            - Benign tumors usually grow slowly and do not spread to other parts of the body.
                            - While benign tumors can sometimes cause issues depending on their size and location, they are generally not life-threatening.
                            """)
                        
                        
                    else:
                        st.error('Prediction failed. Check the log for details.')
            else:
                st.error('Failed to retrieve model input features. Check the log for details.')
        else:
            st.error('Failed to load the model. Check the log for details.')
    else:
        st.error('Failed to download the model from S3. Check the log for details.')

# Run the Streamlit app
if __name__ == '__main__':
    run_app()

