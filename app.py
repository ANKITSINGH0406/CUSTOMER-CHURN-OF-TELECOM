import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lifelines import CoxPHFitter

# --------------------------
# LOAD MODELS
# --------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
cph_model = joblib.load("cph_model.pkl")

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

st.title("📡 Telecom Customer Churn Prediction System")
st.markdown("### ML + Survival Analysis (Cox PH Model)")

# --------------------------
# USER INPUT SECTION
# --------------------------

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

with col2:
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"])

# --------------------------
# PREDICTION
# --------------------------

if st.button("🔍 Predict Churn Risk"):

    # Create base dataframe
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_data])

    # Scale numeric
    input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Ensure all ML columns exist
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    # --------------------------
    # Random Forest Prediction
    # --------------------------

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # --------------------------
    # FIXED CPH Risk Score
    # --------------------------

    # Get exact columns used during CPH training
    cph_columns = cph_model.params_.index.tolist()

    # Create correct input structure
    cph_input = pd.DataFrame(0, index=[0], columns=cph_columns)

    # Fill matching values
    for col in cph_columns:
        if col in input_df.columns:
            cph_input[col] = input_df[col].values[0]

    # Predict hazard
    risk_score = cph_model.predict_partial_hazard(cph_input)
    risk_score_value = float(risk_score.values[0])

    # --------------------------
    # DISPLAY RESULTS
    # --------------------------

    st.subheader("📊 Results")

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.error("⚠ Customer Likely to CHURN")
        else:
            st.success("✅ Customer Likely to STAY")

        st.metric("Churn Probability", f"{probability:.2%}")

    with colB:
        st.metric("Hazard Risk Score (CPH)", f"{risk_score_value:.2f}")

        if risk_score_value > 1:
            st.warning("High churn risk over time")
        else:
            st.info("Low long-term churn risk")
