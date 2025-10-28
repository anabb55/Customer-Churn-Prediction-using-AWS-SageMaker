import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

train = pd.read_csv("data/processed/train_log_regression.csv")
validation = pd.read_csv("data/processed/val_log_regression.csv")
test = pd.read_csv("data/processed/test_log_regression.csv")

X_train, y_train = train.drop(columns=["Churn"]), train["Churn"].values
X_test, y_test = test.drop(columns=["Churn"]), test["Churn"].values

lr = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1, class_weight="balanced")
lr.fit(X_train, y_train)

proba_lr = lr.predict_proba(X_test)[:,1]
pred_lr  = (proba_lr >= 0.5).astype(int)
acc_lr   = accuracy_score(y_test, pred_lr)
cm_lr    = confusion_matrix(y_test, pred_lr)

print("Accuracy: ", acc_lr)
print("Confusion matrix:\n", cm_lr)
