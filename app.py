import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Retention Intelligence System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("feature_engineered_customer_churn.csv")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Retention Intelligence System")
st.write("Predict whether a customer is likely to churn.")
st.caption("AI-powered Machine Learning Dashboard for Customer Churn Prediction")
# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

    monthly = st.slider(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=0.5
    )

with col2:
    total = st.slider(
        "Total Charges ($)",
        min_value=0.0,
        max_value=9000.0,
        value=800.0,
        step=10.0
    )

st.divider()

predict = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)

# -----------------------------
# Predict
# -----------------------------

if predict:

    sample = df.drop(
        ["customerID", "Churn"],
        axis=1,
        errors="ignore"
    ).iloc[[0]].copy()

    sample["tenure"] = tenure
    sample["MonthlyCharges"] = monthly
    sample["TotalCharges"] = total
    sample["AverageMonthlySpend"] = total / max(tenure, 1)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"### Churn Probability: {probability*100:.2f}%")

    st.progress(float(probability))

    st.subheader("Risk Level")

    if probability > 0.80:
        st.error("🔴 High Churn Risk")

    elif probability > 0.50:
        st.warning("🟡 Medium Churn Risk")

    else:
        st.success("🟢 Low Churn Risk")

    st.subheader("Recommendation")

    if probability > 0.80:
        st.warning("✔ Offer 20% Discount")
        st.warning("✔ Assign Dedicated Customer Support")
        st.warning("✔ Provide Loyalty Rewards")

    elif probability > 0.50:
        st.info("✔ Offer Personalized Discounts")
        st.info("✔ Send Reminder Emails")
        st.info("✔ Provide Special Benefits")

    else:
        st.success("✔ Continue Regular Engagement")
        st.success("✔ Maintain Good Customer Service")
        st.success("✔ Reward Loyal Customers")

    st.subheader("Customer Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tenure", f"{tenure} Months")

    with col2:
        st.metric("Monthly Charges", f"${monthly:.2f}")

    with col3:
        st.metric("Total Charges", f"${total:.2f}")

        st.divider()

st.caption("Built with ❤️ using Streamlit | Scikit-learn | Python")