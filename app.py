# ============================================================
# CREDIT CARD FRAUD DETECTION USING DEEP LEARNING
# Streamlit Web Application
# ============================================================

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\gargn\OneDrive\Desktop\AI & ML\Capstone 2\DL Model"
)

# You can override this using the DL_MODEL_DIR environment variable.
ARTIFACT_DIR = Path(
    os.getenv("DL_MODEL_DIR", str(PROJECT_DIR / "models"))
)


# ============================================================
# 3. HELPER: FIND SAVED ARTIFACTS
# ============================================================

def find_artifact(filename):
    """
    Search the configured artifact folder first, then the
    project root folder.
    """
    possible_paths = [
        ARTIFACT_DIR / filename,
        PROJECT_DIR / filename
    ]

    for path in possible_paths:
        if path.exists():
            return path

    searched_paths = "\n".join(str(p) for p in possible_paths)

    raise FileNotFoundError(
        f"Could not find {filename}.\n\n"
        f"Searched these locations:\n{searched_paths}"
    )


# ============================================================
# 4. LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource(show_spinner="Loading the trained DL model...")
def load_artifacts():

    model_path = find_artifact("best_fraud_dl_model.keras")
    preprocessor_path = find_artifact("fraud_preprocessor.joblib")
    metadata_path = find_artifact("fraud_dl_metadata.json")
    threshold_path = find_artifact("fraud_dl_threshold.json")

    # Load trained neural network
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Load preprocessing objects
    preprocessor = joblib.load(preprocessor_path)

    # Load feature metadata
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Load classification threshold
    with open(threshold_path, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)

    # Extract feature names
    feature_columns = metadata.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "The metadata JSON must contain a non-empty "
            "'feature_columns' list."
        )

    # Support common threshold key names
    threshold = None

    if isinstance(threshold_info, dict):
        for key in [
            "threshold",
            "best_threshold",
            "optimal_threshold",
            "decision_threshold"
        ]:
            if key in threshold_info:
                threshold = float(threshold_info[key])
                break

    elif isinstance(threshold_info, (int, float)):
        threshold = float(threshold_info)

    if threshold is None:
        raise ValueError(
            "Could not find the classification threshold "
            "in fraud_dl_threshold.json."
        )

    if not 0 <= threshold <= 1:
        raise ValueError("The saved threshold must be between 0 and 1.")

    return model, preprocessor, metadata, feature_columns, threshold


# ============================================================
# 5. PREPROCESS INPUT DATA
# ============================================================

def preprocess_data(data, preprocessor, feature_columns):
    """
    Apply the same saved preprocessing used during training.

    Expected preprocessor formats:
    1. Dictionary containing 'imputer' and 'scaler'
    2. Dictionary containing a fitted transformer under
       'preprocessor'
    3. A fitted sklearn transformer/pipeline
    """

    # Select features in the exact training order
    data = data.reindex(columns=feature_columns).copy()

    # Convert values to numeric and handle infinities
    for column in feature_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan)

    # Case 1: preprocessing dictionary
    if isinstance(preprocessor, dict):

        # Support a nested preprocessing pipeline
        if "preprocessor" in preprocessor:
            transformer = preprocessor["preprocessor"]
            transformed = transformer.transform(data)

        else:
            imputer = preprocessor.get("imputer")
            scaler = preprocessor.get("scaler")

            if imputer is None or scaler is None:
                raise ValueError(
                    "The preprocessing dictionary must contain "
                    "'imputer' and 'scaler', or a fitted "
                    "'preprocessor' transformer."
                )

            transformed = imputer.transform(data)
            transformed = scaler.transform(transformed)

    # Case 2: a fitted transformer or sklearn Pipeline
    else:
        transformed = preprocessor.transform(data)

    transformed = np.asarray(transformed, dtype=np.float32)

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Preprocessing produced non-finite values. "
            "Check the saved imputer and scaler."
        )

    return transformed


# ============================================================
# 6. GENERATE FRAUD PREDICTIONS
# ============================================================

def predict_fraud(data, model, preprocessor,
                  feature_columns, threshold):

    X_processed = preprocess_data(
        data,
        preprocessor,
        feature_columns
    )

    probabilities = model.predict(
        X_processed,
        verbose=0
    ).reshape(-1)

    if not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid prediction values.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(
            "The model output is not a valid probability. "
            "Check that the saved model has a sigmoid output."
        )

    predicted_classes = (
        probabilities >= threshold
    ).astype(int)

    results = data.copy()

    results["Fraud_Probability"] = probabilities
    results["Predicted_Class"] = predicted_classes

    results["Prediction"] = np.where(
        predicted_classes == 1,
        "Fraudulent Transaction",
        "Legitimate Transaction"
    )

    results["Risk_Level"] = np.select(
        [
            probabilities >= max(threshold, 0.80),
            probabilities >= threshold
        ],
        [
            "High",
            "Flagged"
        ],
        default="Low"
    )

    return results


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select an option",
    [
        "Home",
        "Single Transaction",
        "Batch CSV Prediction",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Deep Learning model for educational and analytical use. "
    "Predictions should not be treated as definitive proof of fraud."
)


# ============================================================
# 8. LOAD ARTIFACTS
# ============================================================

try:
    model, preprocessor, metadata, feature_columns, threshold = (
        load_artifacts()
    )

    artifacts_loaded = True

except Exception as error:
    artifacts_loaded = False

    st.error("The model or its preprocessing files could not be loaded.")

    st.code(str(error))

    st.info(
        "Check that the model, preprocessing file, metadata JSON, "
        "and threshold JSON exist in your models folder or project root."
    )


# ============================================================
# 9. HOME PAGE
# ============================================================

if page == "Home":

    st.title("💳 Credit Card Fraud Detection")
    st.subheader("Deep Learning Classification System")

    st.write(
        """
        This application uses a trained deep learning neural network
        to estimate whether a credit card transaction is potentially
        fraudulent or legitimate.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "Deep Learning")

    with col2:
        st.metric("Classification", "Binary")

    with col3:
        st.metric("Decision Threshold", f"{threshold:.4f}"
                  if artifacts_loaded else "Unavailable")

    st.markdown("---")

    st.subheader("How to use the application")

    st.markdown(
        """
        **1. Single Transaction:** Enter the transaction features to
        receive a prediction.

        **2. Batch CSV Prediction:** Upload a CSV containing the
        transaction feature columns to predict multiple records.

        **3. Model Evaluation:** If your CSV contains actual `Class`
        labels, compare predictions against those labels using
        classification metrics.

        **Class definitions:**
        - `0` — Legitimate transaction
        - `1` — Fraudulent transaction
        """
    )

    if artifacts_loaded:
        st.success("Model and preprocessing artifacts loaded successfully.")


# ============================================================
# 10. SINGLE TRANSACTION PREDICTION
# ============================================================

elif page == "Single Transaction":

    st.title("🔍 Single Transaction Prediction")

    if not artifacts_loaded:
        st.warning("Resolve the artifact-loading error before predicting.")

    else:
        st.write(
            "Enter the features for one transaction. "
            "All feature columns required by the trained model are shown."
        )

        st.warning(
            "The default values are placeholders, not a real transaction. "
            "Replace them with actual feature values for a meaningful prediction."
        )

        with st.form("single_transaction_form"):

            transaction = {}

            # Group features to make the form easier to navigate
            basic_features = [
                column for column in ["Time", "Amount"]
                if column in feature_columns
            ]

            v_features = [
                column for column in feature_columns
                if column not in basic_features
            ]

            st.subheader("Basic Transaction Information")

            basic_cols = st.columns(2)

            for i, column in enumerate(basic_features):
                with basic_cols[i % 2]:
                    transaction[column] = st.number_input(
                        column,
                        value=0.0,
                        format="%.6f",
                        key=f"single_{column}"
                    )

            st.subheader("Additional Model Features")

            if v_features:
                with st.expander(
                    "Enter the remaining features",
                    expanded=True
                ):
                    feature_cols = st.columns(3)

                    for i, column in enumerate(v_features):
                        with feature_cols[i % 3]:
                            transaction[column] = st.number_input(
                                column,
                                value=0.0,
                                format="%.6f",
                                key=f"single_{column}"
                            )

            submitted = st.form_submit_button(
                "Predict Transaction",
                type="primary",
                use_container_width=True
            )

        if submitted:

            input_df = pd.DataFrame(
                [transaction],
                columns=feature_columns
            )

            try:
                result = predict_fraud(
                    input_df,
                    model,
                    preprocessor,
                    feature_columns,
                    threshold
                )

                probability = float(
                    result["Fraud_Probability"].iloc[0]
                )

                predicted_class = int(
                    result["Predicted_Class"].iloc[0]
                )

                st.markdown("---")
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    if predicted_class == 1:
                        st.error("⚠️ Potential Fraud Detected")
                    else:
                        st.success("✅ Predicted Legitimate")

                with col2:
                    st.metric(
                        "Estimated Fraud Probability",
                        f"{probability:.2%}"
                    )

                st.progress(
                    int(round(probability * 100))
                )

                st.write(
                    f"**Decision threshold:** `{threshold:.4f}`"
                )

                st.write(
                    f"**Risk indicator:** {result['Risk_Level'].iloc[0]}"
                )

                st.caption(
                    "This is a model-generated estimate, not a definitive "
                    "determination that a transaction is fraudulent."
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ============================================================
# 11. BATCH CSV PREDICTION
# ============================================================

elif page == "Batch CSV Prediction":

    st.title("📁 Batch Transaction Prediction")

    st.write(
        "Upload a CSV containing the transaction feature columns "
        "used to train your model."
    )

    st.info(
        "If the file contains a `Class` column, it will be retained "
        "for evaluation but will not be used as a prediction feature."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )

            st.write(f"**Rows:** {len(uploaded_df):,}")
            st.write(f"**Columns:** {len(uploaded_df.columns):,}")

            if not artifacts_loaded:
                st.warning(
                    "Resolve the artifact-loading error before predicting."
                )

            else:
                # Keep actual labels separate from model inputs
                actual_labels = None

                if "Class" in uploaded_df.columns:
                    actual_labels = pd.to_numeric(
                        uploaded_df["Class"],
                        errors="coerce"
                    )

                # Identify missing required features
                missing_features = [
                    column for column in feature_columns
                    if column not in uploaded_df.columns
                ]

                if missing_features:
                    st.error(
                        "The uploaded CSV is missing required columns: "
                        + ", ".join(missing_features)
                    )

                    st.write(
                        "Required model features:",
                        feature_columns
                    )

                elif len(uploaded_df) == 0:
                    st.warning("The uploaded CSV contains no transactions.")

                else:
                    input_df = uploaded_df[feature_columns].copy()

                    if st.button(
                        "Run Batch Prediction",
                        type="primary"
                    ):

                        with st.spinner("Predicting transactions..."):

                            results = predict_fraud(
                                input_df,
                                model,
                                preprocessor,
                                feature_columns,
                                threshold
                            )

                            # Preserve actual labels in output
                            if actual_labels is not None:
                                results["Actual_Class"] = (
                                    actual_labels.to_numpy()
                                )

                            st.session_state["fraud_results"] = results
                            st.session_state["fraud_actual_labels"] = (
                                actual_labels
                            )

                    if "fraud_results" in st.session_state:

                        results = st.session_state["fraud_results"]

                        st.markdown("---")
                        st.subheader("Batch Prediction Results")

                        total = len(results)
                        fraud_count = int(
                            (results["Predicted_Class"] == 1).sum()
                        )
                        legitimate_count = total - fraud_count

                        col1, col2, col3 = st.columns(3)

                        col1.metric("Transactions Processed", f"{total:,}")
                        col2.metric("Flagged as Potential Fraud", f"{fraud_count:,}")
                        col3.metric("Predicted Legitimate", f"{legitimate_count:,}")

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_data,
                            file_name="fraud_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as error:
            st.error(f"Could not process the uploaded CSV: {error}")


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write(
        """
        Upload a labelled test dataset containing the required model
        features and a `Class` column. The actual labels are used only
        for evaluation, not as input to the prediction model.
        """
    )

    evaluation_file = st.file_uploader(
        "Upload labelled test CSV",
        type=["csv"],
        key="evaluation_upload"
    )

    if evaluation_file is not None:

        try:
            evaluation_df = pd.read_csv(evaluation_file)

            if not artifacts_loaded:
                st.warning(
                    "Resolve the artifact-loading error before evaluation."
                )

            elif "Class" not in evaluation_df.columns:
                st.error(
                    "The evaluation CSV must contain a `Class` column."
                )

            else:
                missing_features = [
                    column for column in feature_columns
                    if column not in evaluation_df.columns
                ]

                if missing_features:
                    st.error(
                        "Missing required feature columns: "
                        + ", ".join(missing_features)
                    )

                else:
                    y_numeric = pd.to_numeric(
                        evaluation_df["Class"],
                        errors="coerce"
                    )

                    valid_mask = y_numeric.isin([0, 1])

                    if not valid_mask.all():
                        st.warning(
                            f"Skipping {(~valid_mask).sum()} rows with "
                            "missing or invalid Class labels."
                        )

                    evaluation_df = evaluation_df.loc[
                        valid_mask
                    ].copy()

                    y_true = y_numeric.loc[valid_mask].astype(int)

                    if evaluation_df.empty:
                        st.error("No rows with valid binary labels remain.")

                    elif y_true.nunique() < 2:
                        st.error(
                            "Evaluation requires both classes (0 and 1) "
                            "to calculate ROC-AUC and PR-AUC."
                        )

                    elif st.button(
                        "Evaluate Model",
                        type="primary"
                    ):

                        X_eval = evaluation_df[feature_columns]

                        predictions = predict_fraud(
                            X_eval,
                            model,
                            preprocessor,
                            feature_columns,
                            threshold
                        )

                        y_pred = predictions[
                            "Predicted_Class"
                        ].to_numpy()

                        y_prob = predictions[
                            "Fraud_Probability"
                        ].to_numpy()

                        # Core evaluation metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(
                            y_true, y_pred, zero_division=0
                        )
                        recall = recall_score(
                            y_true, y_pred, zero_division=0
                        )
                        f1 = f1_score(
                            y_true, y_pred, zero_division=0
                        )
                        roc_auc = roc_auc_score(y_true, y_prob)
                        pr_auc = average_precision_score(y_true, y_prob)

                        st.markdown("---")
                        st.subheader("Evaluation Metrics")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy", f"{accuracy:.4f}")
                        c2.metric("Precision", f"{precision:.4f}")
                        c3.metric("Recall", f"{recall:.4f}")

                        c4, c5, c6 = st.columns(3)
                        c4.metric("F1 Score", f"{f1:.4f}")
                        c5.metric("ROC-AUC", f"{roc_auc:.4f}")
                        c6.metric("PR-AUC (Average Precision)", f"{pr_auc:.4f}")

                        st.write(
                            f"**Decision threshold used:** `{threshold:.4f}`"
                        )

                        st.markdown("---")
                        st.subheader("Confusion Matrix")

                        cm = confusion_matrix(
                            y_true,
                            y_pred,
                            labels=[0, 1]
                        )

                        fig, ax = plt.subplots(figsize=(6, 4))

                        ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=[
                                "Legitimate (0)",
                                "Fraud (1)"
                            ]
                        ).plot(
                            ax=ax,
                            values_format="d",
                            colorbar=False
                        )

                        ax.set_title("Deep Learning Confusion Matrix")
                        st.pyplot(fig)
                        plt.close(fig)

                        st.markdown("---")
                        st.subheader("Classification Report")

                        report = classification_report(
                            y_true,
                            y_pred,
                            labels=[0, 1],
                            target_names=[
                                "Legitimate",
                                "Fraud"
                            ],
                            output_dict=True,
                            zero_division=0
                        )

                        st.dataframe(
                            pd.DataFrame(report).transpose(),
                            use_container_width=True
                        )

                        st.markdown("---")
                        st.subheader("Prediction Details")

                        output_df = predictions.copy()
                        output_df["Actual_Class"] = y_true.to_numpy()

                        st.dataframe(
                            output_df,
                            use_container_width=True
                        )

                        st.download_button(
                            label="Download Evaluation Predictions",
                            data=output_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="fraud_evaluation_results.csv",
                            mime="text/csv"
                        )

        except Exception as error:
            st.error(f"Evaluation failed: {error}")


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection using Deep Learning | "
    "Academic Capstone Project"
)