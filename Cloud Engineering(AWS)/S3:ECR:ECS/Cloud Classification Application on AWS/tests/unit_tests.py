import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from app import load_data, slider_values

# Sample data for testing
test_df = pd.DataFrame({
    'log_entropy': [0.1, 0.2, 0.3],
    'IR_norm_range': [0.4, 0.5, 0.6],
    'entropy_x_contrast': [0.7, 0.8, 0.9],
    'class': [0, 1, 0]
})

test_file_path = Path("/tmp/test_data.csv")

def test_load_data_file_path():
    # Mock the creation of the test CSV file locally
    test_df.to_csv(test_file_path, index=False)

    # Mock the Path.exists method to return True
    with patch.object(Path, 'exists', return_value=True):
        load_data(test_file_path, "dummy_s3_key")

    # Assert that the file was created in the correct path
    assert test_file_path.exists()

def test_load_data_output_types():
    # Mock the creation of the test CSV file locally
    test_df.to_csv(test_file_path, index=False)

    # Mock the Path.exists method to return True
    with patch.object(Path, 'exists', return_value=True):
        class_labels, features, targets = load_data(test_file_path, "dummy_s3_key")

    # Assert that the returned types are as expected
    assert isinstance(class_labels, list)
    assert isinstance(features, np.ndarray)
    assert isinstance(targets, np.ndarray)

    # Assert the contents of the class labels, features, and targets
    assert class_labels == ['Class 0', 'Class 1']
    np.testing.assert_array_equal(features, test_df[['log_entropy', 'IR_norm_range', 'entropy_x_contrast']].values)
    np.testing.assert_array_equal(targets, test_df['class'].values)

def test_slider_values():
    # Test normal case
    test_series = pd.Series([1, 2, 3, 4, 5])
    min_val, max_val, mean_val = slider_values(test_series)
    assert min_val == 1.0
    assert max_val == 5.0
    assert mean_val == 3.0

    # Test negative values
    test_series = pd.Series([-1, -2, -3, -4, -5])
    min_val, max_val, mean_val = slider_values(test_series)
    assert min_val == -5.0
    assert max_val == -1.0
    assert mean_val == -3.0

    # Test special cases (single entry)
    test_series = pd.Series([10])
    min_val, max_val, mean_val = slider_values(test_series)
    assert min_val == 10.0
    assert max_val == 10.0
    assert mean_val == 10.0

    # Test special cases (NaN)
    test_series = pd.Series([1, 2, np.nan, 4, 5])
    min_val, max_val, mean_val = slider_values(test_series)
    assert min_val == 1.0
    assert max_val == 5.0
    assert mean_val == pytest.approx(3.0)  

    # Test special cases (empty input)
    test_series = pd.Series([])
    min_val, max_val, mean_val = slider_values(test_series)
    assert np.isnan(min_val)
    assert np.isnan(max_val)
    assert np.isnan(mean_val)
