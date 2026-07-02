# ==========================================
# Customer Retention Intelligence System
# Step 4 : Feature Engineering
# ==========================================

import pandas as pd
import numpy as np

# -----------------------------
# Load Clean Dataset
# -----------------------------
df = pd.read_csv("cleaned_customer_churn.csv")

print("Dataset Shape :", df.shape)

# ----------------------------------------
# Feature 1 : Tenure Group
# ----------------------------------------
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"],
    include_lowest=True
)

# ----------------------------------------
# Feature 2 : Monthly Charge Category
# ----------------------------------------
df["ChargeCategory"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 120],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# ----------------------------------------
# Feature 3 : Average Monthly Spend
# ----------------------------------------
df["AverageMonthlySpend"] = (
    df["TotalCharges"] / (df["tenure"] + 1)
)

# ----------------------------------------
# Feature 4 : Long Term Customer
# ----------------------------------------
df["LongTermCustomer"] = np.where(
    df["tenure"] >= 24,
    "Yes",
    "No"
)

# ----------------------------------------
# Feature 5 : High Value Customer
# ----------------------------------------
median_charge = df["MonthlyCharges"].median()

df["HighValueCustomer"] = np.where(
    df["MonthlyCharges"] >= median_charge,
    "Yes",
    "No"
)

# ----------------------------------------
# Feature 6 : Total Services Used
# ----------------------------------------
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["TotalServices"] = 0

for col in service_columns:
    df["TotalServices"] += df[col].apply(
        lambda x: 0 if x in ["No", "No internet service"] else 1
    )

# ----------------------------------------
# Check New Features
# ----------------------------------------
print("\nNew Features Added")
print(df[[
    "TenureGroup",
    "ChargeCategory",
    "AverageMonthlySpend",
    "LongTermCustomer",
    "HighValueCustomer",
    "TotalServices"
]].head())

# ----------------------------------------
# One-Hot Encoding
# ----------------------------------------
categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "ChargeCategory",
    "LongTermCustomer",
    "HighValueCustomer"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# ----------------------------------------
# Encode Target Variable
# ----------------------------------------
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# ----------------------------------------
# Dataset Information
# ----------------------------------------
print("\nDataset Shape After Feature Engineering :", df.shape)

print("\nColumns")
print(df.columns)

print("\nFirst 5 Rows")
print(df.head())

# ----------------------------------------
# Save Dataset
# ----------------------------------------
df.to_csv("feature_engineered_customer_churn.csv", index=False)

print("\nFeature Engineering Completed Successfully!")
print("Saved as : feature_engineered_customer_churn.csv")