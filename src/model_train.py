# ==========================================
# Customer Retention Intelligence System
# Step 6 : Model Training
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

# -----------------------------------------
# Load Dataset
# -----------------------------------------

df = pd.read_csv("feature_engineered_customer_churn.csv")

# -----------------------------------------
# Features & Target
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
# Models
# -----------------------------------------

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

}

# -----------------------------------------
# Train Models
# -----------------------------------------

print("\nModel Accuracy")
print("-" * 40)

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name} : {accuracy:.4f}")