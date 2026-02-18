import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lifelines import CoxPHFitter

# Load models
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
cph_model = pickle.load(open("cph_model.pkl", "rb"))

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

    # Add missing columns
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
    # CPH Risk Score
    # --------------------------

    cph_input = input_df.copy()
    cph_input["duration"] = tenure
    cph_input["event"] = 0

    risk_score = cph_model.predict_partial_hazard(cph_input)

    # --------------------------
    # DISPLAY RESULTS
    # --------------------------

    st.subheader("📊 Results")

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.error(f"⚠ Customer Likely to CHURN")
        else:
            st.success(f"✅ Customer Likely to STAY")

        st.metric("Churn Probability", f"{probability:.2%}")

    with colB:
        st.metric("Hazard Risk Score (CPH)", f"{float(risk_score):.2f}")

        if float(risk_score) > 1:
            st.warning("High churn risk over time")
        else:
            st.info("Low long-term churn risk")
