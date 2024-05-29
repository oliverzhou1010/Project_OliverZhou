APP_NAME=cloud-app
ECR_URL=public.ecr.aws/g0i4r9p4
AWS_REGION=us-east-2

requirements.txt: poetry.lock
	poetry export -f requirements.txt --output requirements.txt --without-hashes

requirements: requirements.txt

build: Dockerfile
	docker build -t ${APP_NAME}:latest .

run:
	docker run -d --name ${APP_NAME} \
		-v ~/.aws:/root/.aws \
		-e AWS_PROFILE \
		-p 80:80 ${APP_NAME}

stop:
	docker stop ${APP_NAME} || true
	docker rm ${APP_NAME} || true

login:
	aws ecr-public get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URL}

tag:
	docker tag ${APP_NAME}:latest ${ECR_URL}/${APP_NAME}:latest

push:
	docker push ${ECR_URL}/${APP_NAME}:latest

publish: build tag push

clean: stop
