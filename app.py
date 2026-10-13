
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = Path(
    "fraud_detection_artifacts/fraud_detection_model.joblib"
)

# -----------------------------
# Load saved model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)

st.title("🛡️ Fraud Detection AI")
st.caption(
    "Machine Learning powered credit card transaction risk assessment"
)

if not MODEL_PATH.exists():
    st.error(
        "Model file nahi mili. Check karo ke "
        "fraud_detection_artifacts folder aur model file mojood hain."
    )
    st.stop()

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    threshold = float(artifacts["threshold"])
    feature_names = artifacts["feature_names"]
except Exception as exc:
    st.error(f"Model load nahi hua: {exc}")
    st.stop()

if len(feature_names) != 30:
    st.error(
        f"Expected 30 features, lekin {len(feature_names)} mile. "
        "Saved artifact aur notebook check karo."
    )
    st.stop()

# -----------------------------
# Sidebar information
# -----------------------------
with st.sidebar:
    st.header("Model Information")
    st.write("Algorithm: Logistic Regression")
    st.write(f"Decision threshold: {threshold:.2f}")
    st.write(f"Input features: {len(feature_names)}")
    st.warning(
        "Educational demo only. Isay real financial decisions "
        "ke liye bina proper validation use na karein."
    )

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter Transaction Features")

st.info(
    "Dataset ke V1–V28 features PCA-transformed hain. "
    "Accurate prediction ke liye original dataset ki feature values "
    "use karo. Neeche demo ke liye numerical inputs diye gaye hain."
)

with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)

    input_values = {}

    for index, feature in enumerate(feature_names):
        column = [col1, col2, col3][index % 3]

        with column:
            if feature == "Time":
                value = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    help="Seconds elapsed since the first transaction."
                )
            elif feature == "Amount":
                value = st.number_input(
                    feature,
                    min_value=0.0,
                    value=100.0,
                    help="Transaction amount."
                )
            else:
                value = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

        input_values[feature] = value

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_df = pd.DataFrame(
        [[input_values[name] for name in feature_names]],
        columns=feature_names
    )

    try:
        scaled_input = scaler.transform(input_df)
        fraud_probability = float(
            model.predict_proba(scaled_input)[0, 1]
        )

        prediction = int(fraud_probability >= threshold)

        st.divider()
        st.subheader("Analysis Result")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Estimated Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

        with metric_col2:
            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%"
            )

        if prediction == 1:
            st.error(
                "⚠️ FRAUD ALERT — Transaction ko model ne "
                "potentially fraudulent flag kiya hai."
            )
        else:
            st.success(
                "✅ LEGITIMATE CLASSIFICATION — Model ne "
                "transaction ko fraud threshold se neeche score kiya."
            )

        st.progress(fraud_probability)

        st.caption(
            "Prediction model ka estimate hai, guarantee nahi. "
            "Fraud probability ko confirmed fraud ya financial loss "
            "ka proof na samjhein."
        )

    except Exception as exc:
        st.error(f"Prediction ke waqt error aaya: {exc}")
