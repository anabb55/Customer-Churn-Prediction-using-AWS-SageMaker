# Customer-Churn-Prediction-using-AWS-SageMaker

## Project Overview

The main idea behind this project was to explore the capabilities of **AWS Cloud** for developing and deploying a machine learning model. 
The goal was to predict customer churn (whether a customer will leave a service) and to compare a cloud-trained **XGBoost model** with a locally trained **Logistic Regression model** using the same dataset.

## Data Preprocessing

- Removed column customerID
- Converted TotalCharges to numeric and filled missing values with 0
- Encoded binary columns (Yes/No, Male/Female) as 0/1
- Applied one-hot encoding to categorical features
- Scaled numerical columns using StandardScaler
- Split data into Train (80%), Validation (10%), and Test (10%) sets
- Creating separate dataset versions for Logistic Regression and XGBoost models

## Model Training and Comparison

  After preprocessing, all processed datasets were uploaded to **Amazon S3**, where they served as inputs for model training and evaluation.

  ### XGBoost - Trained and Deployed on AWS SageMaker
  - Used AWS SageMaker to train the XGBoost model directly in the cloud.
  - Configured the training job with defined S3 input paths, hyperparameters, and compute instance (ml.m5.large).
  - After training, the model was deployed as a SageMaker endpoint, enabling batch predictions.
 
 
  ### Logistic Regression – Trained Locally
  - Trained locally using scikit-learn on the same preprocessed data.
  - The model served as a baseline for performance comparison with the cloud-trained XGBoost model.
  - Evaluated on the same test dataset to ensure fairness.
  
  ### Results Comparison

   | Model                   | Accuracy | AUC      |
   | ----------------------- | -------- | -------- |
   | **XGBoost**             | **0.81** | **0.87** |
   | **Logistic Regression** |  0.75    | 0.84     |

   ## Conclusion

   The XGBoost model achieved slightly better results than Logistic Regression, mainly due to its ability to capture non-linear relationships and feature interactions.

   Working with **AWS SageMaker** and **Amazon S3** showed me how essential cloud technologies are for **Data Science** and **Data Engineering** today - enabling scalable data processing, model training, and deployment that would be impossible to achieve locally.

   ## Exploratory Data Analysis (EDA)

   EDA was conducted to better understand the dataset before modeling.
   
   Below are the key visualizations and insights:

1. **Feature Categorization**

   Separated features into:

   -Cateorical features

   -Numerical features

   -Encoded categorical features(SeniorCitizen)

2. **Correlation Analysis**

   Created a heatmap to visualize correlations among all numerical features.

   ![Correlation Heatmap](images/Figure_1.png)

   **Finding:** Strong correlation between TotalCharges, MonthlyCharges, and tenure as expected (since TotalCharges ≈ MonthlyCharges × tenure).

3. **Target Variable Distribution (Churn)**

   ![Correlation Heatmap](images/Figure_2.png)

   **Finding:** There is a noticeable imbalance between churn and non-churn classes - most customers did not churn, which may cause model bias if not handled.

4. **Monthly Charges by Churn**

    Visualized how Monthly Charges differ between churned and non-churned customers.

   ![Correlation Heatmap](images/Figure_3.png)

   **Finding:** Customers who churned generally had higher monthly charges.

   I also computed **Spearman correlation**: 0.1847 -> Positive correlation between higher monthly charges and churn probability.

5. **Tenure by Churn**

   ![Correlation Heatmap](images/Figure_4.png)

   **Finding:** Customers with shorter tenure are more likely to churn.

    **Spearman correlation**:  -0.3671. The correlation is negative, therefore longer tenure leads to lower churn risk.

6. **Gender-Based Churn Distribution**

   ![Correlation Heatmap](images/Figure_5.png)

   ![Correlation Heatmap](images/Figure_6.png)

   **Finding:** Both genders show similar churn proportions - gender has little influence on churn likelihood.




   
   
   
   
   

   



