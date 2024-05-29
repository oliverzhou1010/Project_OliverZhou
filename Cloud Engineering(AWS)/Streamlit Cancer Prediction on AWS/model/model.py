import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import pickle
import boto3
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a Random Forest model and upload results to S3.')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')

args = parser.parse_args()

# Load hyperparameters from the provided config file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

n_estimators = config.get('n_estimators', 100)
max_depth = config.get('max_depth', None)
max_features = config.get('max_features', 'auto')

# Load data
data = pd.read_csv('s3://projectsiyanli/breast-cancer.csv')

# Split data
X = data.drop(['diagnosis', 'id'], axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Compute confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Save model
model_path = 'model_B.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

# Save confusion matrix and classification report
conf_matrix_path = 'confusion_matrix.txt'
class_report_path = 'classification_report.txt'

with open(conf_matrix_path, 'w') as f:
    f.write(str(conf_matrix))

with open(class_report_path, 'w') as f:
    f.write(class_report)

# Upload files to S3
s3 = boto3.client('s3')
bucket_name = 'projectsiyanli'

s3.upload_file(model_path, bucket_name, model_path)
s3.upload_file(conf_matrix_path, bucket_name, conf_matrix_path)
s3.upload_file(class_report_path, bucket_name, class_report_path)

# Print confusion matrix and classification report
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

