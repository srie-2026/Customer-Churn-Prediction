# Machine Learning Assignment 2: Customer Churn Prediction

## a. Problem statement
A telecom company loses revenue every time a customer cancels the service, and winning a new customer costs much more than keeping an existing one. The goal of this project is to predict, from a customer's profile, services and billing details, whether the customer will **churn** (leave) or stay. Five classification models are trained on the same data, compared on the same held-out test set with six evaluation metrics, and demonstrated in an interactive Streamlit app. The aim is to identify likely churners early so that the business can target retention offers at them.

## b. Dataset description
- **Source:** IBM Telco Customer Churn dataset, a public dataset on Kaggle: [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Instances:** 7,043 customers. After a stratified split (25% test, `random_state=48`): **5,282 for training** and **1,761 for testing** (`test_data.csv`).
- **Features:** the original file has 21 columns. The identifier `customerID` was dropped, which leaves **19 input features** and 1 target column.
  - *Customer profile:* `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - *Account and billing:* `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - *Subscribed services:* `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target variable:** `Churn` (0 = customer stayed, 1 = customer left).
- **Class balance:** imbalanced, about 73.5% No churn and 26.5% Yes churn (5,174 vs 1,869 customers). The test set keeps the same ratio (1,294 vs 467). A model that always predicts "No churn" would already score about **73.5% accuracy**, so accuracy alone is not enough, and AUC, F1 and MCC are also used.
- **Preprocessing:** `TotalCharges` is stored as text in the raw file, so it was converted to numbers; the 11 blank values (new customers) were filled with the median. Churn was mapped from No/Yes to 0/1. Categorical columns are one-hot encoded and numeric columns are scaled inside a scikit-learn `Pipeline`, so the saved models accept raw feature columns directly.

## c. Github Repository Link
[Customer Churn Prediction on GitHub](https://github.com/srie-2026/Customer-Churn-Prediction)

The repository contains: `app.py` (Streamlit app), `requirements.txt` (pinned versions), `README.md`, `test_data.csv` (held-out test data), `model/` (the five saved models in `.joblib` format), the training notebook `ML_Assignment.ipynb` and the original `dataset/`.

## Live Streamlit App Link
[Customer Churn Prediction App](https://customer-churn-prediction-sri.streamlit.app/)

Upload `test_data.csv`, compare all five models in one table, then pick a model to see its metrics, confusion matrix and classification report. The app loads the saved models and does not retrain.

## d. Models used
Five classifiers were trained on the same training split: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=6`), K-Nearest Neighbors (`k=5`), Gaussian Naive Bayes, and Support Vector Machine (probability estimates enabled so that AUC can be computed). All metrics below are for the churn class (class 1) on the 1,761 test records.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7973 | 0.8364 | 0.6403 | 0.5375 | 0.5844 | 0.4547 |
| **Decision Tree** | 0.7825 | 0.8110 | 0.6055 | 0.5161 | 0.5572 | 0.4166 |
| **KNN** | 0.7632 | 0.7752 | 0.5573 | 0.5203 | 0.5382 | 0.3796 |
| **Naive Bayes** | 0.6917 | 0.8117 | 0.4561 | 0.8458 | 0.5926 | 0.4255 |
| **Support Vector Machine** | 0.7939 | 0.7974 | 0.6529 | 0.4754 | 0.5502 | 0.4297 |

Errors on the 467 actual churners and 1,294 loyal customers in the test set (from the confusion matrices):

| ML Model Name | Churners caught | Churners missed | Loyal customers wrongly flagged |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 251 | 216 (46.3%) | 141 (10.9%) |
| Decision Tree | 241 | 226 (48.4%) | 157 (12.1%) |
| KNN | 243 | 224 (48.0%) | 193 (14.9%) |
| Naive Bayes | 395 | 72 (15.4%) | 471 (36.4%) |
| Support Vector Machine | 222 | 245 (52.5%) | 118 (9.1%) |

## e. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Best on accuracy (0.7973), AUC (0.8364) and MCC (0.4547), and second-best on precision (0.6403). Its errors are well balanced: it catches 251 of 467 churners with only 141 false alarms. Churn appears to depend mostly on tenure, contract type and monthly charges in a roughly linear way, which suits this model. |
| **Decision Tree** | Slightly below Logistic Regression on every metric (accuracy 0.7825, AUC 0.8110, MCC 0.4166). Limiting the depth to 6 keeps it from overfitting and keeps the rules readable, but it still misses 226 churners (48.4%) and gives up some flexibility. |
| **KNN** | Weakest model on AUC (0.7752), F1 (0.5382) and MCC (0.3796). One-hot encoding turns the many Yes/No service columns into many binary dimensions, which makes distances between customers less informative, and it raises 193 false alarms. |
| **Naive Bayes** | Lowest accuracy (0.6917), which is even below the 73.5% "always predict No churn" baseline, and the lowest precision (0.4561). But it has by far the highest recall (0.8458, only 72 churners missed) and the highest F1 (0.5926), at the price of 471 false alarms. Its AUC (0.8117) is decent, so it ranks customers well but its decision threshold leans towards predicting churn. The independence assumption does not hold for correlated service columns. Useful only if missing a churner is much costlier than a wasted offer. |
| **Support Vector Machine** | Second-best accuracy (0.7939) and the highest precision (0.6529) with the fewest false alarms (118), so a churn prediction from it is the most trustworthy. However, it has the lowest recall (0.4754) and misses 245 of 467 churners (52.5%), and its AUC (0.7974) is below Logistic Regression, Naive Bayes and the Decision Tree. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It has the highest accuracy, AUC and MCC, balanced errors, and it is the simplest and fastest model to train and explain. Naive Bayes would only be preferred if the business wanted to catch as many churners as possible regardless of false alarms. |
