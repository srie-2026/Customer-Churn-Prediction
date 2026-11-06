# Machine Learning Assignment 2: Customer Churn Prediction

## a. Problem statement
The objective of this project is to build and evaluate multiple machine learning classification models to predict customer churn in a telecommunications company. By identifying customers who are likely to leave, the business can take proactive retention measures.

## b. Dataset description
This project uses the **IBM Telco Customer Churn dataset**. 
- **Total Instances:** 7,043 records (split into 5,282 for training and 1,761 for testing).
- **Features:** 20 columns (19 input features and 1 target column). Features include customer demographics (e.g., SeniorCitizen, Partner, Dependents), account details (e.g., tenure, Contract, MonthlyCharges, TotalCharges), and subscribed services (e.g., InternetService, TechSupport, StreamingTV).
- **Target Variable:** `Churn` (Binary: 0 for No Churn, 1 for Yes Churn).

## c. Github Repository Link
[Customer Churn Prediction on GitHub](https://github.com/srie-2026/Customer-Churn-Prediction)

## Live Streamlit App Link
[Customer Churn Prediction App](https://customer-churn-prediction-sri.streamlit.app/)

## d. Models used
Five classification models were trained and evaluated. The performance metrics on the test data are summarized below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7973 | 0.8364 | 0.6403 | 0.5375 | 0.5844 | 0.4547 |
| **Decision Tree** | 0.7825 | 0.8110 | 0.6055 | 0.5161 | 0.5572 | 0.4166 |
| **KNN** | 0.7632 | 0.7752 | 0.5573 | 0.5203 | 0.5382 | 0.3796 |
| **Naive Bayes** | 0.6917 | 0.8117 | 0.4561 | 0.8458 | 0.5926 | 0.4255 |
| **Support Vector Machine** | 0.7939 | 0.7974 | 0.6529 | 0.4754 | 0.5502 | 0.4297 |

## e. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed the best overall, achieving the highest accuracy (79.73%) and AUC (83.64%), indicating a strong ability to distinguish between churn and non-churn classes. |
| **Decision Tree** | Showed moderate performance with 78.25% accuracy. It successfully captured non-linear patterns but underperformed Logistic Regression slightly. |
| **KNN** | Demonstrated lower performance across most metrics (76.32% accuracy), likely due to the dataset's mix of categorical and numerical features. |
| **Naive Bayes** | Yielded the lowest accuracy (69.17%) and precision, but achieved the highest recall (84.58%). It is highly effective at identifying actual churners but produces many false positives. |
| **Support Vector Machine** | Delivered very strong accuracy (79.39%) and the highest precision (65.29%), making it highly reliable when predicting that a customer will churn, though its recall was the lowest. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall winner due to its balance of the highest accuracy, highest AUC, and strong MCC score. |