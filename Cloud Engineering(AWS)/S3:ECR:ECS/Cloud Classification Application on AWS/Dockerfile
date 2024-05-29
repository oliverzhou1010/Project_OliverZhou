# Use the official Python image as the base image and ensure that we target x86_64 architectures
FROM --platform=linux/x86_64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version
RUN /usr/local/bin/python -m pip install --upgrade pip

# Copy the requirements file and install the dependencies without caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the working directory
COPY . .

# Expose port 80 for HTTP traffic
EXPOSE 80

# Ensure all Python output is sent directly to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit application
CMD ["streamlit", "run", "--server.port=80", "--browser.serverAddress=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false", "app.py"]
