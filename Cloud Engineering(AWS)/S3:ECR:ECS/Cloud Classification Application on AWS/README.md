###  Running the Docker 

Choosing the port: 

- **Port 80**: Commonly used for HTTP traffic and doesn't require specifying a port in the URL (e.g., http://localhost).
- **Port 8501**: Default port for Streamlit when running locally without specifying a port.

   ```bash
   docker build -t cloud-app .

   docker run --name cloud-app --platform linux/amd64 -v ~/.aws:/root/.aws:ro -v ${PWD}/artifacts:/app/data -p 80:80 -d cloud-app
   
   docker run --platform linux/amd64 -p 8080:80 -v "$(pwd)/artifacts:/app/artifacts" cloud-app

### Setting up ECR/push images to repo

1. 
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/r0r7a4d3

2. 
cloud_app % docker build --platform linux/x86_64 -t cloud-app .

3.
docker tag cloud-app:latest public.ecr.aws/r0r7a4d3/cloud-app:latest

4.
docker push public.ecr.aws/r0r7a4d3/cloud-app:latest


Then setting up clusters on ECS.

Then Create Task Definition.

Then Delopy -> Create Service.



