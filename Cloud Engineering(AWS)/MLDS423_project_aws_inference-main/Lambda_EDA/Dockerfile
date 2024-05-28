FROM public.ecr.aws/lambda/python:3.8

# Copy function code
COPY main.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install -r requirements.txt
ENV MPLCONFIGDIR=/tmp


# Set the CMD to your handler
CMD [ "main.lambda_handler" ]
