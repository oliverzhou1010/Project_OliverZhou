# EDA via AWS ECR and Lambda

## Setup and Usage

Clone the repository

```
git clone https://github.com/MSIA/MLDS423_project_aws_inference.git
```

To set up the environment to run this project on a local machine:

```
python -m venv .venv
source .venv/bin/activate
```

Setup AWS credential

```
aws sso login --profile <your_aws_profile_name>
export AWS_PROFILE=<your_aws_profile_name>
```

# Push Dockerfile to AWS ECR 
```
aws ecr get-login-password --region aws-region | docker login --username AWS --password-stdin aws.account.dkr.ecr.aws.region.amazonaws.com

aws ecr create-repository --repository-name <ecr_repo_name>

docker build -t <docker_name> .

docker tag <docker_name>:latest aws.account.ecr.aws.accout.amazonaws.com/<ecr_repo_name>:latest

docker push aws.account.dkr.ecr.aws.region.amazonaws.com/<ecr_repo_name>:latest
```

