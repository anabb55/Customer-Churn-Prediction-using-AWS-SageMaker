import os
import time
import boto3 
import sagemaker 
import sklearn 
import pandas as pd


sm_boto3 =  boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'churn-bucket-sagemaker'
print("Using bucket " + bucket)
