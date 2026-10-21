import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.title("Credit Card Fraud Detection & Risk Analytics")
st.caption("Explainable AI • XGBoost • Threshold Optimization • Business Cost Analysis")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_final.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

THRESHOLD = config["threshold"]
FEATURES = config["features"]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH).drop_duplicates().copy()

st.divider()
st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    fraud_count = int(df["Class"].sum())
    st.metric("Fraud Transactions", f"{fraud_count:,}")

with col3:
    fraud_rate = df["Class"].mean() * 100
    st.metric("Fraud Rate", f"{fraud_rate:.3f}%")

with col4:
    total_value = df["Amount"].sum()
    st.metric("Total Transaction Value", f"${total_value:,.2f}")

st.divider()
st.subheader("Fraud Analytics")

amount_bins = [0, 1, 10, 50, 100, 500, np.inf]
amount_labels = ["0-1", "1-10", "10-50", "50-100", "100-500", "500+"]

df["Amount_Bucket"] = pd.cut(
    df["Amount"],
    bins=amount_bins,
    labels=amount_labels,
    include_lowest=True
)

amount_summary = (
    df.groupby("Amount_Bucket", observed=False)["Class"]
    .agg(Transaction_Count="count", Fraud_Count="sum")
    .reset_index()
)

amount_summary["Fraud_Rate_%"] = (
    amount_summary["Fraud_Count"] / amount_summary["Transaction_Count"] * 100
)

st.dataframe(amount_summary, use_container_width=True, hide_index=True)

st.bar_chart(
    amount_summary.set_index("Amount_Bucket")["Fraud_Rate_%"],
    y_label="Fraud Rate (%)"
)
st.divider()
st.subheader("Fraud Rate by Time Period")

time_bins = pd.cut(
    df["Time"],
    bins=6,
    labels=["Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Period 6"]
)

time_summary = (
    df.assign(Time_Period=time_bins)
    .groupby("Time_Period", observed=False)["Class"]
    .agg(Transaction_Count="count", Fraud_Count="sum")
    .reset_index()
)

time_summary["Fraud_Rate_%"] = (
    time_summary["Fraud_Count"] / time_summary["Transaction_Count"] * 100
)

st.dataframe(time_summary, use_container_width=True, hide_index=True)

st.bar_chart(
    time_summary.set_index("Time_Period")["Fraud_Rate_%"],
    y_label="Fraud Rate (%)"
)
st.divider()
st.subheader("Model Performance")
metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Precision", "92.5%")

with metric2:
    st.metric("Recall", "77.9%")

with metric3:
    st.metric("F1 Score", "84.6%")

with metric4:
    st.metric("PR-AUC", "83.1%")

st.write("Confusion Matrix")

confusion_matrix = pd.DataFrame(
    [[56645, 6], [21, 74]],
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"]
)

st.dataframe(confusion_matrix, use_container_width=True)
st.info(
    "At the selected threshold, the model identifies 74 of 95 fraud transactions "
    "while generating only 6 false alerts. This gives a recall of 77.9% and "
    "precision of 92.5% on the untouched test set."
)
st.divider()
st.subheader("Threshold & Business Cost Analysis")

threshold_data = pd.DataFrame({
    "Threshold Strategy": ["F1-optimal", "Cost-optimal"],
    "Threshold": [0.3879, 0.1089],
    "Validation Precision": [98.46, 94.29],
    "Validation Recall": [84.21, 86.84],
    "Validation F1": [90.78, 90.41],
    "Relative Cost": [121, 104]
})

st.dataframe(
    threshold_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Cost uses an illustrative scenario score: false positive = 1 and "
    "false negative = 10. These are relative business assumptions, "
    "not actual monetary losses."
)
st.divider()
st.subheader("Explainable AI — Global Feature Importance")

shap_importance = pd.DataFrame({
    "Feature": ["V4", "V12", "V10", "V14", "V11", "V19", "V8", "V3", "Amount", "V21"],
    "Mean_Abs_SHAP": [
        1.114189, 0.849733, 0.823256, 0.796635, 0.301775,
        0.253577, 0.232128, 0.226617, 0.223871, 0.217977
    ]
})

st.dataframe(
    shap_importance,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    shap_importance.set_index("Feature")["Mean_Abs_SHAP"],
    y_label="Mean |SHAP value|"
)

st.caption(
    "Global SHAP importance summarizes how strongly each feature contributes "
    "to the model's predictions on the test set. It explains model behavior, "
    "not causality."
)
st.subheader("Example Fraud Alert Explanation")

st.write("Selected transaction")

example_transaction = pd.DataFrame({
    "Metric": [
        "Actual Class",
        "Predicted Probability",
        "Decision Threshold",
        "Model Decision"
    ],
    "Value": [
        "Fraud",
        "76.4%",
        "10.9%",
        "Flagged as Fraud"
    ]
})

st.dataframe(
    example_transaction,
    use_container_width=True,
    hide_index=True
)
st.write("Top factors influencing this prediction")

local_explanation = pd.DataFrame({
    "Feature": ["V14", "V10", "V12", "V28", "V26", "V4", "V13", "V8", "V7", "Amount"],
    "Feature Value": [
        -3.823, -3.539, -3.993, 0.396, -0.278,
        2.213, -1.937, 1.197, -2.356, 76.94
    ],
    "SHAP Contribution": [
        5.731, 1.792, 1.164, -0.913, -0.593,
        0.550, 0.521, -0.509, 0.448, -0.444
    ]
})

st.dataframe(
    local_explanation,
    use_container_width=True,
    hide_index=True
)


st.divider()
st.subheader("Project Summary")
summary1, summary2 = st.columns(2)

with summary1:
    st.markdown(
        """
        ### 🔄 Workflow

        Data Quality → SQL Analytics → EDA → Preprocessing →  
        Baseline Models → Imbalance Handling → XGBoost →  
        Threshold Optimization → Business Cost → SHAP
        """
    )

with summary2:
    st.markdown(
        """
        ### 🤖 Final Model

        **XGBoost**

        Validation-selected threshold: **0.1089**
        """
    )

summary3, summary4 = st.columns(2)

with summary3:
    st.markdown(
        """
        ### 📊 Test Performance

        **Precision:** 92.5%  
        **Recall:** 77.9%  
        **F1:** 84.6%  
        **PR-AUC:** 83.1%
        """
    )

with summary4:
    st.markdown(
        """
        ### ⚠️ Limitations

        V1–V28 are anonymized/PCA-transformed predictors.

        The business cost framework uses illustrative scenario
        scores rather than actual financial losses.
        """
    )