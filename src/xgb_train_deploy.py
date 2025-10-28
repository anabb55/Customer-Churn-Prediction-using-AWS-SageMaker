import os
import pandas as pd
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
import boto3
from sagemaker.serializers import CSVSerializer
from sklearn.metrics import  accuracy_score, confusion_matrix
import numpy as np

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



test = pd.read_csv("data/processed/test_xgb.csv", header=None)
y_test = test.iloc[:, 0].to_numpy()
X_test = test.iloc[:, 1:]

def predict_in_batches(predictor, X, batch_size = 200):
    probs = []
    for i in range(0, X.shape[0], batch_size):
        batch = X.iloc[i:i+batch_size].values
        out = predictor.predict(batch)

        if isinstance(out, (bytearray, bytes)):
            out = out.decode("utf-8")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        probs.extend([float(v) for v in lines])

    return np.array(probs, dtype=float)

y_pred_proba = predict_in_batches(xgb_predictor, X_test)

y_pred = (y_pred_proba >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy: ", acc)
print("Confusion matricx: \n", cm)

