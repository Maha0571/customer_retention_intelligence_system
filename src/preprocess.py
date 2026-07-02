# ==========================================
# Customer Retention Intelligence System
# Step 5 : Data Preprocessing
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Feature Engineered Dataset
# -----------------------------
df = pd.read_csv("feature_engineered_customer_churn.csv")

print("Dataset Shape :", df.shape)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(["customerID", "Churn"], axis=1, errors="ignore")
y = df["Churn"]

print("Features Shape :", X.shape)
print("Target Shape :", y.shape)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape :", X_train.shape)
print("Test Shape :", X_test.shape)

# -----------------------------
# Standard Scaling
# -----------------------------
scaler = StandardScaler()

numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns

X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

print("\nData Preprocessing Completed Successfully!")