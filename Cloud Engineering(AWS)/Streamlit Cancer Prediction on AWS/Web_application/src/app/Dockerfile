# Use the official Python image as the base image and ensure that we target x86_64 architectures
FROM --platform=linux/x86_64 python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Update pip to use the latest version
RUN /usr/local/bin/python -m pip install --upgrade pip
# RUN pip install --upgrade pip

# Copy the requirements file to the working directory
COPY . .

# Install the required dependencies
RUN pip install -r requirements.txt

# Expose port 80 for http traffic
EXPOSE 80

ENV PYTHONUNBUFFERED=1

# Set the command to run the Streamlit application
CMD ["streamlit", "run", "--server.port", "80", "--browser.serverAddress","0.0.0.0","--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", "app.py"]

