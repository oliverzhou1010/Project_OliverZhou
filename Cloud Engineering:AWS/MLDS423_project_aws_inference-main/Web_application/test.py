import pytest
from unittest.mock import patch, MagicMock
import logging
import boto3
# Import the functions from the hw3_app script
from src.app.app import download_model_from_s3, predict, get_model_input_features

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Test download_model_from_s3 function
def test_download_model_from_s3():
    mock_s3 = MagicMock()

    # Successful download
    mock_s3.download_file.return_value = True
    result = download_model_from_s3('fake_bucket', 'fake_key', 'fake_path')
    assert result == None

# Test predict function
def test_predict():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]

    input_data = [[-2.30, 0.03, 8.43,2.69,11.43]]
    result = predict(input_data, 15, mock_model)
    assert result == [0]

    # Exception during prediction
    mock_model.predict.side_effect = Exception('Test Exception')
    result = predict(input_data,15, mock_model)
    assert result is None

# Test get_model_input_features function
def test_get_model_input_features():
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ['log_entropy', 'IR_norm_range','entropy_x_contrast','visible_second_angular_momentum','Vis_norm_range']

    result = get_model_input_features(mock_model)
    assert result == ['log_entropy', 'IR_norm_range','entropy_x_contrast','visible_second_angular_momentum','Vis_norm_range']

    # Model does not have feature_names_in_ attribute
    del mock_model.feature_names_in_
    result = get_model_input_features(mock_model)
    assert result is None


# Test download_model_from_s3 function
def test_download_model_from_s3_1():
    mock_s3 = MagicMock()

    # Successful download
    mock_s3.download_file.return_value = True
    result = download_model_from_s3('fake_bucket_1', 'fake_key_1', 'fake_path_1')
    assert result == None

# Test predict function
def test_predict_1():
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]

    input_data = [[-2.31, 0.04, 8.44, 2.70, 11.44]]
    result = predict(input_data, mock_model)
    assert result == [1]

    # Exception during prediction
    mock_model.predict.side_effect = Exception('Test Exception 1')
    result = predict(input_data, mock_model)
    assert result is None

# Test get_model_input_features function
def test_get_model_input_features_1():
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ['log_entropy_1', 'IR_norm_range_1', 'entropy_x_contrast_1', 'visible_second_angular_momentum_1', 'Vis_norm_range_1']

    result = get_model_input_features(mock_model)
    assert result == ['log_entropy_1', 'IR_norm_range_1', 'entropy_x_contrast_1', 'visible_second_angular_momentum_1', 'Vis_norm_range_1']

    # Model does not have feature_names_in_ attribute
    del mock_model.feature_names_in_
    result = get_model_input_features(mock_model)
    assert result is None