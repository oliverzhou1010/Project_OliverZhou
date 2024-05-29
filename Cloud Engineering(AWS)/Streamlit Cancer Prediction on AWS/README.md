# Breast Cancer: Prediction and Visualization via AWS platform

Authors: Jason Huang, Wesley Wang, Siyan Li, Xinyang (Oliver) Zhou

Date: Spring 2024

Master of Science in Machine Learning and Data Science 

Northwestern University

### Abstract

This project aims to leverage the breast cancer dataset, consisting of 569 instances with 32 attributes, to build a prediction and visualization portal for individuals concerned about breast cancer and medical practitioners. The dataset includes features such as mean radius, texture, perimeter, area, smoothness, compactness, concavity, and several other characteristics of the cell nuclei present in the images.

### Lambda_EDA:
The data was collected and processed using AWS Lambda functions, which captured the data from an S3 bucket. The data was then loaded into an EC2 cluster for training machine learning models.

- main.py: The main script to orchestrate the workflow, including data processing and model training.
- Dockerfile: Contains instructions to build the Docker image for EDA (Exploratory Data Analysis) using AWS Lambda.
- project_lambda_2.yaml: Configuration file for AWS Lambda functions used in the project.
- requirements.txt: Lists the Python dependencies needed for the project.

### model:
The EC2 cluster was used to train various machine learning models to predict the likelihood of breast cancer. The training process involved extensive data preprocessing, feature selection, and model tuning to ensure high accuracy and reliability.

- model.py: Contains the code for defining and training the machine learning model.
- config.yaml: Configuration file for setting up and training the machine learning model.

### Web_application/src/app:
The trained models were deployed using a web application built with Streamlit. The application runs on an ECS cluster, and the entire deployment process is managed using Docker containers for consistency and scalability.

- app.py: The main application script for running the Streamlit web interface.
- Dockerfile: Defines the steps to build the Docker image for the Streamlit web application.
- requirements.txt: Lists the Python dependencies needed to run the web application.

### Docker Usage for Team Members with AWS Permissions

Step 1: Configure AWS Keys
The keys will be stored in ~/.aws/credentials.
```
aws configure
```

```
AWS Access Key ID: [input access key]
AWS Secret Access Key: [input secret access key]
Default region name [us-east-2]: [press Enter]
Default output format [None]: [press Enter]
```

Export personal credential
```
export AWS_ACCESS_KEY_ID=[input access key]
export AWS_SECRET_ACCESS_KEY=[input secret access key]
```

Step 2: Build and Run Docker Containers
```
docker build -t demo -f Dockerfile .
```

```
docker run demo
```
