
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# --------------------------------------------------
# 2. LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    model_files = {
        "Logistic Regression": "logistic_regression_model.joblib",
        "Decision Tree": "decision_tree_model.joblib",
        "KNN": "knn_model.joblib",
        "Naive Bayes": "naive_bayes_model.joblib",
        "Support Vector Machine": "svm_model.joblib"
    }

    loaded_models = {}

    for model_name, file_name in model_files.items():
        model_path = MODEL_DIR / file_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("📊 Customer Churn Prediction System")

st.markdown("""
### Machine Learning Classification Dashboard

This application evaluates five trained machine learning
classifiers using the Telco Customer Churn dataset, shared
on Kaggle by BlastChar and attributed there to IBM Sample Data Sets.

Upload your test CSV file to compare model performance.
""")

st.divider()

# --------------------------------------------------
# 4. LOAD MODELS
# --------------------------------------------------

try:
    models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --------------------------------------------------
# 5. UPLOAD TEST DATA
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.write("Dataset Preview")
        st.dataframe(data.head(), width="stretch")

        # --------------------------------------------------
        # 6. DATA VALIDATION
        # --------------------------------------------------

        target_column = "Churn"

        if target_column not in data.columns:
            st.error(
                "The uploaded CSV must contain the Churn target column."
            )
            st.stop()

        # Separate features and target
        y_test = data[target_column].copy()
        X_test = data.drop(columns=[target_column])

        # Support both encoded and original target values
        if y_test.dtype == "object":
            y_test = y_test.astype(str).str.strip().map({
                "No": 0,
                "Yes": 1,
                "0": 0,
                "1": 1
            })

        if y_test.isna().any():
            st.error("Invalid or missing values found in Churn target.")
            st.stop()

        y_test = y_test.astype(int)

        if not set(y_test.unique()).issubset({0, 1}):
            st.error("Churn target must contain only 0 and 1.")
            st.stop()

        if y_test.nunique() < 2:
            st.error(
                "The test dataset must contain both churn classes "
                "(0 and 1) to calculate AUC."
            )
            st.stop()

        # Get expected feature columns from the saved pipeline
        first_model = next(iter(models.values()))
        expected_features = list(first_model.feature_names_in_)

        missing_features = [
            col for col in expected_features
            if col not in X_test.columns
        ]

        if missing_features:
            st.error(
                f"Missing required features: {missing_features}"
            )
            st.stop()

        # Keep exactly the training feature columns in correct order
        X_test = X_test[expected_features]

        st.write("Test Dataset Information")

        col1, col2 = st.columns(2)

        col1.metric("Total Test Records", len(X_test))
        col2.metric("Input Features", len(X_test.columns))

        st.divider()

        # --------------------------------------------------
        # 7. EVALUATE ALL FIVE MODELS
        # --------------------------------------------------

        st.subheader("Model Performance Comparison")

        results = {}
        predictions = {}

        for model_name, model in models.items():

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            predictions[model_name] = y_pred

            results[model_name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, y_pred, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

        results_df = pd.DataFrame(results).T

        st.dataframe(
            results_df.style.format("{:.4f}"),
            width="stretch"
        )

        # --------------------------------------------------
        # 8. SELECT INDIVIDUAL MODEL
        # --------------------------------------------------

        st.divider()

        st.subheader("Individual Model Evaluation")

        selected_model = st.selectbox(
            "Select a Machine Learning Model",
            list(models.keys())
        )

        selected_prediction = predictions[selected_model]

        selected_metrics = results[selected_model]

        st.write(f"### {selected_model} Results")

        # Display six metrics
        metric_cols = st.columns(3)

        for i, (metric_name, metric_value) in enumerate(
            selected_metrics.items()
        ):
            metric_cols[i % 3].metric(
                metric_name,
                f"{metric_value:.4f}"
            )

        # --------------------------------------------------
        # 9. CONFUSION MATRIX
        # --------------------------------------------------
        st.divider()

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(figsize=(2.6, 2.0), dpi=100)

        cm = confusion_matrix(
            y_test,
            selected_prediction,
            labels=[0, 1]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Churn", "Churn"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        ax.set_title("Confusion Matrix", fontsize=6)
        ax.set_xlabel("Predicted Label", fontsize=5)
        ax.set_ylabel("True Label", fontsize=5)
        ax.tick_params(axis="both", labelsize=5)

        plt.tight_layout()

        st.pyplot(fig, width="content")

        plt.close(fig)

        
        # --------------------------------------------------
        # 10. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            selected_prediction,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.4f}"),
            width="stretch"
        )

        # --------------------------------------------------
        # 11. FINAL OBSERVATION
        # --------------------------------------------------

        st.divider()

        st.subheader("Overall Best Performing Model")

        best_model = results_df["Accuracy"].idxmax()

        st.success(
            f"Based on test accuracy, the best performing model "
            f"is {best_model} with accuracy "
            f"{results_df.loc[best_model, 'Accuracy']:.4f}."
        )

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")

else:

    st.info(
        "Please upload your test_data.csv file to begin model evaluation."
    )

    st.markdown("""
    **Instructions:**
    1. Upload the test CSV generated from the ML training notebook.
    2. Select a classifier to view its detailed evaluation.
    3. Compare all five classifiers using the metrics table.
    """)
