import pandas as pd
import os 
#os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
print('Just for file testing')

print("Uploading file")
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# connect to s3 AWS profile
s3_client = boto3.client('s3')

def load_data(bucket_name, file_key):
    # Download the file from S3
    local_file_path = '/tmp/<your_dataset>'
    s3_client.download_file(bucket_name, file_key, local_file_path)
    return pd.read_csv(local_file_path)

def preprocess_data(data):
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def visualize_data(data, directory, bucket_name=None):
    fig_paths = []

    # Basic statistics
    stats = data.describe().transpose()
    stats_path = os.path.join(directory, 'Basic_Statistics.csv')
    stats.to_csv(stats_path)
    logger.info("Basic statistics saved: Basic_Statistics.csv")

    # Correlation matrix with selected variables and diagnosis
    selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[selected_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    fig_path = os.path.join(directory, 'correlation_matrix.png')
    plt.savefig(fig_path)
    fig_paths.append(fig_path)
    logger.info("EDA Figure saved: correlation_matrix.png")

    # Distribution of diagnosis
    plt.figure(figsize=(10, 6))
    data['diagnosis'].value_counts().plot(kind='bar', color='lightblue')
    plt.title('Distribution of Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    fig_path = os.path.join(directory, 'diagnosis_distribution.png')
    plt.savefig(fig_path)
    fig_paths.append(fig_path)
    logger.info("EDA Figure saved: diagnosis_distribution.png")

    # Distribution of radius mean
    plt.figure(figsize=(10, 6))
    sns.histplot(data['radius_mean'], kde=True, bins=30, color='lightblue')
    plt.title('Distribution of Radius Mean')
    fig_path = os.path.join(directory, 'radius_mean_distribution.png')
    plt.savefig(fig_path)
    fig_paths.append(fig_path)
    logger.info("EDA Figure saved: radius_mean_distribution.png")

    # Distribution of area mean
    plt.figure(figsize=(10, 6))
    sns.histplot(data['area_mean'], kde=True, bins=30, color='lightblue')
    plt.title('Distribution of Area Mean')
    fig_path = os.path.join(directory, 'area_mean_distribution.png')
    plt.savefig(fig_path)
    fig_paths.append(fig_path)
    logger.info("EDA Figure saved: area_mean_distribution.png")

    # Pair plot
    sns.pairplot(data[selected_features], hue='diagnosis', palette='coolwarm')
    plt.suptitle('Pair Plot of Selected Features', y=1.02)
    fig_path = os.path.join(directory, 'pair_plot.png')
    plt.savefig(fig_path)
    fig_paths.append(fig_path)
    logger.info("EDA Figure saved: pair_plot.png")

    # Upload to S3 if bucket_name is provided
    if bucket_name:
        s3_client = boto3.client('s3')
        s3_path_prefix = '<project_folder>/'
        s3_client.upload_file(stats_path, bucket_name, s3_path_prefix + 'Basic_Statistics.csv')
        for path in fig_paths:
            s3_client.upload_file(path, bucket_name, s3_path_prefix + os.path.basename(path))
            logger.info(f"EDA Figure uploaded to S3: {s3_path_prefix + os.path.basename(path)}")

    return fig_paths

def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    file_key = event['file_key']
    directory = '/tmp'

    # Load data
    data = load_data(bucket_name, file_key)
    logger.info("Data loaded successfully")

    # Preprocess data
    data = preprocess_data(data)
    logger.info("Data preprocessed successfully")

    # Visualize data
    fig_paths = visualize_data(data, directory, bucket_name)
    logger.info(f"Figures saved: {fig_paths}")

    return {
        'statusCode': 200,
        'body': 'Data processed and visualizations saved successfully.'
    }

if __name__ == "__main__":
    # Test locally
    class Context:
        pass

    event = {
        'bucket_name': '<your_bucket>',
        'file_key': '<project_folder>/<your_dataset>'
    }
    context = Context()
    lambda_handler(event, context)

