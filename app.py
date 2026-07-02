import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("feature_engineered_customer_churn.csv")

st.title("Customer Retention Intelligence System")

st.write("Predict whether a customer is likely to churn.")

# -----------------------------
# User Inputs
# -----------------------------
tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=0.50
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0,
    step=10.0
)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):

    sample = df.drop(
        ["customerID", "Churn"],
        axis=1,
        errors="ignore"
    ).iloc[[0]].copy()

    sample["tenure"] = tenure
    sample["MonthlyCharges"] = monthly
    sample["TotalCharges"] = total
    sample["AverageMonthlySpend"] = total / (tenure + 1)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

    st.write(f"Churn Probability : {probability*100:.2f}%")

    # -----------------------------
    # Retention Suggestion
    # -----------------------------
    st.subheader("Recommendation")

if probability > 0.80:
    st.error("High Churn Risk")
    st.write("The customer is very likely to leave the company.")
    st.warning("Offer a special discount.")
    st.warning("Assign dedicated customer support.")
    st.warning("Provide loyalty rewards to retain the customer.")

elif probability > 0.50:
    st.warning("Medium Churn Risk")
    st.write("The customer has a moderate chance of leaving the company.")
    st.info("Offer an upgrade plan.")
    st.info("Provide free technical support.")
    st.info("Engage with the customer through follow-up offers.")

else:
    st.success("Low Churn Risk")
    st.write("The customer is likely to stay with the company.")
    st.success("No immediate retention action is required.")