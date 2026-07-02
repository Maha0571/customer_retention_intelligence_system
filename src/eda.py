# ==========================================
# Customer Retention Intelligence System
# Step 3 : Exploratory Data Analysis (EDA)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Clean Dataset
# -----------------------------
df = pd.read_csv("cleaned_customer_churn.csv")

# Style
sns.set(style="whitegrid")

# -----------------------------
# Dataset Shape
# -----------------------------
print("Dataset Shape :", df.shape)

# -----------------------------
# Target Variable Distribution
# -----------------------------
print("\nChurn Distribution")
print(df["Churn"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")
plt.show()

# -----------------------------
# Senior Citizen Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="SeniorCitizen", data=df)
plt.title("Senior Citizen Distribution")
plt.show()

# -----------------------------
# Contract Type Distribution
# -----------------------------
plt.figure(figsize=(7,4))
sns.countplot(x="Contract", data=df)
plt.title("Contract Type")
plt.xticks(rotation=15)
plt.show()

# -----------------------------
# Payment Method
# -----------------------------
plt.figure(figsize=(10,5))
sns.countplot(x="PaymentMethod", data=df)
plt.title("Payment Method")
plt.xticks(rotation=30)
plt.show()

# -----------------------------
# Internet Service
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="InternetService", data=df)
plt.title("Internet Service")
plt.show()

# -----------------------------
# Monthly Charges Distribution
# -----------------------------
plt.figure(figsize=(8,4))
plt.hist(df["MonthlyCharges"], bins=30)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Customers")
plt.show()

# -----------------------------
# Tenure Distribution
# -----------------------------
plt.figure(figsize=(8,4))
plt.hist(df["tenure"], bins=30)
plt.title("Tenure Distribution")
plt.xlabel("Months")
plt.ylabel("Customers")
plt.show()

# -----------------------------
# Churn vs Gender
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.show()

# -----------------------------
# Churn vs Contract
# -----------------------------
plt.figure(figsize=(8,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract vs Churn")
plt.xticks(rotation=15)
plt.show()

# -----------------------------
# Churn vs Internet Service
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Internet Service vs Churn")
plt.show()

# -----------------------------
# Boxplot
# Monthly Charges vs Churn
# -----------------------------
plt.figure(figsize=(7,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# -----------------------------
# Boxplot
# Tenure vs Churn
# -----------------------------
plt.figure(figsize=(7,4))
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully")