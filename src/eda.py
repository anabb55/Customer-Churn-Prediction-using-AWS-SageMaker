import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RAW = os.path.join("data", "telco_churn.csv")

def main():
     if not os.path.exists(RAW):
        raise SystemExit("Dataset not found")
     
     data = pd.read_csv(RAW)
     if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
     data = data.fillna(0)


     missing_rows = data[data.isna().any(axis=1)]
     
     categorical_data = data.select_dtypes(include=['object']).columns.tolist()
     numerical_data = data.select_dtypes(exclude=['object']).columns.tolist()

     categorical_enc_features = ['SeniorCitizen']
     
     ex = ['customerID', 'Churn']
     categorical_features = [feature for feature in categorical_data if feature not in ex]

     ex = ['SeniorCitizen']
     numerical_features = [feature for feature in numerical_data if feature not in ex]

     print("Categorical Features:", categorical_features,'\n')
     print("Numerical Features:", numerical_features,'\n')
     print("Categorical Encoded Features:", categorical_enc_features,'\n')

     print(data[numerical_features].describe())
     

     ##assumption: there is correlation between total charges and monthly charges and tenure
     corr_num = data[numerical_features].corr()
     plt.figure(figsize=(10, 8))
     sns.heatmap(corr_num, annot=True, cmap='coolwarm')
     plt.show()

     ## total cahrges should primarily be determined by the product of Monthly chages and tenure
     data_new = data[numerical_features].copy()
     data_new['MonthlyCharges_x_tenure'] = data_new['MonthlyCharges'] * data_new['tenure']
     correlation = data_new[['TotalCharges', 'MonthlyCharges_x_tenure']].corr().iloc[0,1]
     print("Correlation between TotalCharges and MonthlyCharges_x_tenure is: ", correlation)

     dist_churn = data['Churn'].value_counts()
     labels = dist_churn.index
     counts = dist_churn.values

     plt.figure(figsize=(6, 6))
     plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['blue', 'red'])
     plt.title('Churn Distribution')
     plt.axis('equal')  
     plt.show()

    ## There is an imbalance between churn and non-churn classes. This situation may result in bias during prediction, as the prediction results will tend to be accurate for the majority class.

    ##Monthly charges by churn

     colors = ["#7e92e1", "#df472c"]
     plt.figure(figsize=(8,6))
     sns.histplot(data = data, x='MonthlyCharges', hue='Churn', palette=colors)

     plt.title('Distribution of Monthly Charges by Churn')
     plt.xlabel('Monthly Charges')
     plt.ylabel('Number of Customers')
     plt.show()

     mapping = {'Yes': 1, 'No': 0}
     data_corr = data.copy()
     data_corr['Churn'] = data_corr['Churn'].map(mapping)
     corr,_ = spearmanr(data_corr['MonthlyCharges'], data_corr['Churn'])

     print(f"Spearman Correlation between MonthlyCharges and Churn: {corr:.4f}")

     ##Tenure vs Churn
     plt.figure(figsize=(8,6))
     sns.histplot(data = data, x='tenure', hue='Churn', palette=colors)

     plt.title('Distribution of Tenure by Churn')
     plt.xlabel('Tenure(Months)')
     plt.ylabel('Number of Customers')
     plt.show()

     corr,_ = spearmanr(data_corr['tenure'], data_corr['Churn'])

     print(f"Spearman Correlation between MonthlyCharges and Churn: {corr:.4f}")


     ##Churn distribution by Male and Female
     male_churn = data[data['gender'] == 'Male']['Churn'].value_counts(normalize=True)
     female_churn = data[data['gender'] == 'Female']['Churn'].value_counts(normalize=True)

     labels_male = male_churn.index
     counts_male = male_churn.values
     labels_female = female_churn.index
     counts_female = female_churn.values

     plt.figure(figsize=(6, 6))
     plt.pie(counts_male, labels=labels_male, autopct='%1.1f%%', colors=['blue', 'red'])
     plt.title('Churn Distribution by Male')
     plt.axis('equal')  
     plt.show()

     plt.figure(figsize=(6, 6))
     plt.pie(counts_female, labels=labels_female, autopct='%1.1f%%', colors=['blue', 'red'])
     plt.title('Churn Distribution by Female')
     plt.axis('equal')  
     plt.show()




if __name__ == "__main__":
    main()
