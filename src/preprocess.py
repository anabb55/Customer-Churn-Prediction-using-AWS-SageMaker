import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

RAW = os.path.join("data", "telco_churn.csv")
OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(RAW):
        raise SystemExit("Dataset not found")
    
    data = pd.read_csv(RAW).copy()
    data = data.drop(columns=['customerID'], axis=1)
    
   
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.fillna(0)


    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
    for col in binary_cols:
        if data[col].dtype == 'object':
            if col == 'gender':
                data[col] = data[col].map({'Male': 0, 'Female': 1})
            else:
                data[col] = data[col].map({'Yes': 1, 'No': 0})


    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'Churn']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    ##Removing TotalCharges for Logistic Regression model
    data = data.drop(columns = ['TotalCharges'])

    numerical_cols = [ col for col in ['tenure', 'MonthlyCharges', 'TotalCharges'] if col in data.columns]
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    
    bool_cols = [col for col in data.columns if data[col].dtype == 'bool']
    data[bool_cols] = data[bool_cols].astype(int)

    y = data["Churn"]
    X = data.drop(columns=["Churn"])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_reg = pd.concat([y_train.rename('Churn'), X_train], axis=1)
    val_reg   = pd.concat([y_val.rename("Churn"),   X_val],   axis=1)
    test_reg  = pd.concat([y_test.rename('Churn'),  X_test],  axis=1)
    train_reg.to_csv(os.path.join(OUT_DIR, "train_log_regression.csv"), index=False)
    val_reg.to_csv(os.path.join(OUT_DIR, "val_log_regression.csv"), index=False)
    test_reg.to_csv(os.path.join(OUT_DIR, "test_log_regression.csv"), index=False)


    train_xgb = pd.concat([y_train, X_train], axis=1)
    val_xgb   = pd.concat([y_val,   X_val],   axis=1)
    test_xgb  = pd.concat([y_test,  X_test],  axis=1)
    train_xgb.to_csv(os.path.join(OUT_DIR, "train_xgb.csv"), index=False, header=False)
    val_xgb.to_csv(os.path.join(OUT_DIR, "val_xgb.csv"),   index=False, header=False)
    test_xgb.to_csv(os.path.join(OUT_DIR, "test_xgb.csv"), index=False, header=False)

    print("Saved:", os.listdir(OUT_DIR))



if __name__ == "__main__":
    main()