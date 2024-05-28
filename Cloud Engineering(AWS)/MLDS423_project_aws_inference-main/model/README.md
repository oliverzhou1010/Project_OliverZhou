### Our modeling reads data from AWS S3, trained in EC2, and save the model as well as the evaluation metrics back to AWS S3.
### We employ random forest classifier for this task, along with two possible hyperparameter configs "config.yaml" and "config2.yaml".
### To run the model, in the command line run:
  python model.py --config <config_path>
