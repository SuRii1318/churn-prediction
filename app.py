# app.py – Bank Customer Churn Risk Calculator

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="Churn Risk Scorer", layout="wide")
st.title("🏦 Bank Customer Churn Risk Calculator")

# ----------------------------------------------
# 1. Load the trained pipeline
# ----------------------------------------------
model = None
preprocessor = None
pipeline = None

# Try loading a single pipeline file first (recommended)
if os.path.exists('churn_pipeline.pkl'):
    with open('churn_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    st.sidebar.success("✅ Model loaded successfully")
elif os.path.exists('churn_model.pkl') and os.path.exists('preprocessor.pkl'):
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    st.sidebar.success("✅ Model loaded successfully")
else:
    st.sidebar.warning("⚠️ No model found – using demo predictions")

# ----------------------------------------------
# 2. Sidebar – Customer profile inputs
# ----------------------------------------------
st.sidebar.header("📋 Customer Profile")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# ----------------------------------------------
# 3. Derived features (keep in sync with training)
# ----------------------------------------------
balance_salary_ratio = balance / (estimated_salary + 1)
products_per_tenure = num_products / (tenure + 1)
active_products = is_active * num_products
age_tenure = age * tenure

# Build input DataFrame with all columns used during training
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary],
    'BalanceSalaryRatio': [balance_salary_ratio],
    'ProductsPerTenure': [products_per_tenure],
    'Active_Products': [active_products],
    'Age_Tenure': [age_tenure]
})

# ----------------------------------------------
# 4. Prediction
# ----------------------------------------------
if pipeline is not None:
    proba = pipeline.predict_proba(input_data)[0, 1]
elif model is not None and preprocessor is not None:
    transformed = preprocessor.transform(input_data)
    proba = model.predict_proba(transformed)[0, 1]
else:
    # Demo heuristic (fallback)
    proba = 0.3 + (age - 20) * 0.003 - is_active * 0.15 + (1 - has_cr_card) * 0.05
    proba = max(0.0, min(1.0, proba))

# ----------------------------------------------
# 5. Display results
# ----------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Probability", f"{proba:.2%}")
    if proba > 0.5:
        st.error("🔴 High Risk – action recommended")
    else:
        st.success("🟢 Low Risk – likely to stay")

with col2:
    # Probability bar
    fig, ax = plt.subplots()
    ax.bar(["Retain", "Churn"], [1 - proba, proba], color=['green', 'red'])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ----------------------------------------------
# 6. Feature Importance (placeholder – replace with real SHAP values)
# ----------------------------------------------
st.subheader("📊 Top Churn Drivers")
st.markdown("_Replace this chart with your actual SHAP or model‑based importance values._")

# Example importance – update with your real data
features = ['Age', 'Balance', 'NumOfProducts', 'IsActiveMember', 'CreditScore']
importances = [0.25, 0.20, 0.18, 0.15, 0.12]

fig2, ax2 = plt.subplots()
ax2.barh(features, importances, color='skyblue')
ax2.set_xlabel("Relative Importance")
st.pyplot(fig2)