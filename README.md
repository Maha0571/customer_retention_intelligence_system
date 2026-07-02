# Customer Retention Intelligence System

## Project Overview

Customer Retention Intelligence System is a Machine Learning project that predicts whether a customer is likely to churn based on customer information. It also displays the churn probability and a simple retention recommendation.

---

## Problem Statement

Customer churn affects business growth and revenue. This project helps identify customers who are likely to leave so that businesses can take retention actions.

---

## Dataset

IBM Telco Customer Churn Dataset

**Target Variable:** Churn

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit

---

## Project Workflow

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Model Training
* Model Evaluation
* Save Best Model
* Streamlit Deployment

---

## Machine Learning Models

The following models were trained and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Random Forest was selected as the final model because it provided the best overall performance.

---

## Features

* Predict Customer Churn
* Display Churn Probability
* Show Customer Retention Recommendation
* Interactive Streamlit Application

---

## Project Structure

```text
Customer-Retention-System/
│── data/
│── data_cleaning.py
│── eda.py
│── feature_engineering.py
│── preprocessing.py
│── model_training.py
│── model_evaluation.py
│── save_model.py
│── app.py
│── best_model.pkl
│── scaler.pkl
│── README.md
```

---

## Business Benefits

* Identify customers who are likely to churn
* Help businesses improve customer retention
* Support better business decisions

---

## Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Lalitha Maha**
