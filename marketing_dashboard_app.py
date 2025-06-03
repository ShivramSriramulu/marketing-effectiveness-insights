import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Marketing Mix & Attribution Dashboard", layout="wide")
st.title("📈 Marketing ROI Intelligence Dashboard")

# Load data and models
df = pd.read_csv("marketing_data_full.csv", parse_dates=["DATE"])
mmm_model = joblib.load("mmm_model.joblib")
mta_model = joblib.load("mta_model.joblib")
scaler = joblib.load("scaler.joblib")

# === ROI Section ===
st.header("💰 Weekly ROI Trend")

df["total_spend"] = df[["tv_S", "search_S", "facebook_S", "newsletter"]].sum(axis=1)
df["roi"] = df["revenue"] / df["total_spend"]
df["roi_rolling"] = df["roi"].rolling(window=4).mean()

fig_roi, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["DATE"], df["roi"], label="Weekly ROI", alpha=0.5)
ax.plot(df["DATE"], df["roi_rolling"], label="4-Week Avg ROI", color="green")
ax.set_title("ROI Over Time")
ax.set_ylabel("ROI")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig_roi)

# === MMM Coefficients ===
st.header("📊 Marketing Channel Effectiveness (MMM Coefficients)")

mmm_features = ["tv_S", "search_S", "facebook_S", "newsletter"]
mmm_coef = pd.Series(mmm_model.params[1:], index=mmm_features)

fig_coef, ax = plt.subplots()
mmm_coef.sort_values().plot(kind="barh", ax=ax, color="orange")
ax.set_title("Marketing Mix Model Coefficients")
st.pyplot(fig_coef)

# === MTA Prediction ===
st.header("🧠 Conversion Likelihood Prediction (MTA)")

with st.form("predict_form"):
    tv = st.slider("TV Spend", 0, 100_000, 20_000)
    search = st.slider("Search Spend", 0, 50_000, 10_000)
    fb = st.slider("Facebook Spend", 0, 50_000, 10_000)
    newsletter = st.slider("Newsletter Spend", 0, 50_000, 10_000)
    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([[tv, search, fb, newsletter]], columns=["tv_S", "search_S", "facebook_S", "newsletter"])
    input_scaled = scaler.transform(input_data)
    pred = mta_model.predict(input_scaled)[0]
    proba = mta_model.predict_proba(input_scaled)[0][1]
    
    st.markdown(f"### 🔮 Predicted Conversion Likelihood: `{proba:.2%}`")
    st.success("High Conversion Probability!" if proba > 0.5 else "Low Conversion Probability")

# === Footer ===
st.caption("Built by Shivram Sriramulu • Powered by MMM + MTA + Streamlit")
