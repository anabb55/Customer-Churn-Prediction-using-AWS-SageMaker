import boto3
import os

region = 'eu-central-1'
bucket = "churn-bucket-sagemaker-proj"
prefix = "telco-churn/data"

session = boto3.session.Session(region_name=region)
s3 = session.client("s3")


files_to_upload = [
    "train_xgb.csv",
    "val_xgb.csv",
    "test_xgb.csv",
    "train_log_regression.csv",
    "val_log_regression.csv",
    "test_log_regression.csv"
]

for fname in files_to_upload:
    local_path = os.path.join("data", "processed", fname)
    s3_path = f"{prefix}/{fname}"
    s3.upload_file(local_path, bucket, s3_path)