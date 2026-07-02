# ==========================================
# Customer Retention Intelligence System
# Data Cleaning
# ==========================================

import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first rows
print(df.head())

# Dataset Information
print(df.info())

# Shape
print("Shape :", df.shape)

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# ------------------------------------------
# Convert TotalCharges to Numeric
# ------------------------------------------

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Missing values after conversion
print("\nMissing After Conversion")
print(df.isnull().sum())

# Fill Missing TotalCharges
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ------------------------------------------
# Remove Duplicates
# ------------------------------------------

duplicates = df.duplicated().sum()
print("Duplicate Rows :", duplicates)

df = df.drop_duplicates()

# ------------------------------------------
# Check Data Types
# ------------------------------------------

print("\nData Types")
print(df.dtypes)

# ------------------------------------------
# Basic Statistics
# ------------------------------------------

print(df.describe())

# ------------------------------------------
# Unique Values
# ------------------------------------------

for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}")
    print(df[col].unique())

# ------------------------------------------
# Standardize Column Names
# ------------------------------------------

df.columns = df.columns.str.strip()

# ------------------------------------------
# Check Target Variable
# ------------------------------------------

print(df["Churn"].value_counts())

# ------------------------------------------
# Save Cleaned Dataset
# ------------------------------------------

df.to_csv("cleaned_customer_churn.csv", index=False)

print("\nData Cleaning Completed Successfully!")