import os
import pandas as pd
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
import boto3
from sagemaker.serializers import CSVSerializer

role = "arn:aws:iam::715195481214:role/role-churn-sagemaker"
bucket = "churn-bucket-sagemaker-proj"
prefix = "telco-churn/data"
region = "eu-central-1"

boto_sess = boto3.Session(region_name=region)
sess = sagemaker.Session(boto_session=boto_sess)



train_s3 = f"s3://{bucket}/{prefix}/train_xgb.csv"
val_s3  = f"s3://{bucket}/{prefix}/val_xgb.csv"

xgb_image = image_uris.retrieve("xgboost", region, "1.7-1")


xgb = sagemaker.estimator.Estimator(image_uri=xgb_image, role=role, instance_count=1, instance_type="ml.m5.large",  output_path=f"s3://{bucket}/{prefix}/output", sagemaker_session=sess, hyperparameters={
    "objective": "binary:logistic",
        "eval_metric": "auc",
        "num_round": "200",
        "max_depth": "5",
        "gamma": "4"
         
},)

xgb.fit({
    "train": TrainingInput(train_s3, content_type="text/csv"),
    "validation": TrainingInput(val_s3, content_type="text/csv")
})

xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type="ml.m5.large", serializer=CSVSerializer())

xgb_predictor.delete_endpoint()
