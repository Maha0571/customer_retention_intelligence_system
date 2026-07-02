# ==========================================
# Customer Retention Intelligence System
# Step 9 : Save Best Model
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------
# Load Dataset
# -----------------------------------------

df = pd.read_csv("feature_engineered_customer_churn.csv")

# -----------------------------------------
# Features and Target
# -----------------------------------------

X = df.drop(["customerID", "Churn"], axis=1, errors="ignore")
y = df["Churn"]

# -----------------------------------------
# Train Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# Standard Scaling
# -----------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------
# Train Best Model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------
# Save Model & Scaler
# -----------------------------------------

joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully!")
print("Files Created:")
print("1. best_model.pkl")
print("2. scaler.pkl")