import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RAW = os.path.join("data", "telco_churn.csv")

def main():
    if not os.path.exists(RAW):
        raise SystemExit("Dataset not found")
    
    data = pd.read_csv(RAW)
    data = data.drop(columns=['customerID'], axis=1)
    
    
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.fillna(0)


    if "Churn" not in data.columns:
        raise SystemExit("Column Churm can't be found")
    data["Churn"] = (data["Churn"].astype(str).str.lower() == "yes").astype(int)

    pd.get_dummies(data, drop_first=True)


    print(data.shape)
    print(data.head())

    y = data["Churn"].values
    X = data.drop(columns=["Churn"]).values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    main()