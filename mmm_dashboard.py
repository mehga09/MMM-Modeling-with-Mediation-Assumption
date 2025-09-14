# mmm_dashboard_v5.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Page Config & Style
# -------------------------------
st.set_page_config(page_title="📊 MMM Dashboard Pro", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #ff7f24; }
h1 { color: #2E4053; font-weight:bold; }
h2, h3 { color: #1F618D; font-weight:bold; }
.stSlider > div > div > div > div { background: #E67E22 !important; }
.stButton>button { background-color: #4CAF50; color:white; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Marketing Mix Modeling (MMM) Dashboard Pro")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("marketing_dataset.csv", parse_dates=["week"])
df = df.sort_values("week").reset_index(drop=True)

# Fill missing spends with 0
spend_cols = ["facebook_spend","google_spend","tiktok_spend","instagram_spend","snapchat_spend"]
df[spend_cols] = df[spend_cols].fillna(0)

# Lag features for Google spend
df["google_lag1"] = df["google_spend"].shift(1).fillna(0)
df["google_lag2"] = df["google_spend"].shift(2).fillna(0)

# Total Social Spend
df["social_spend"] = df[["facebook_spend","tiktok_spend","instagram_spend","snapchat_spend"]].sum(axis=1)

# Interaction: social spend influencing Google
df["social_to_google"] = df["social_spend"] * df["google_lag1"]

# Week-of-year for seasonality
df["week_of_year"] = df["week"].dt.isocalendar().week
df["month"] = df["week"].dt.to_period("M")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Model Settings")
alpha_val = st.sidebar.slider("Ridge Alpha (Regularization)", 0.0, 10.0, 1.0, 0.1)

feature_options = [
    "facebook_spend","tiktok_spend","instagram_spend","snapchat_spend",
    "google_spend","google_lag1","google_lag2",
    "social_spend","social_to_google",
    "social_followers","average_price","promotions","emails_send","sms_send","week_of_year"
]

selected_features = st.sidebar.multiselect(
    "Select Features for Modeling",
    options=feature_options,
    default=feature_options
)

# Month filter
months = df["month"].astype(str).unique()
selected_months = st.sidebar.multiselect("Select Months", months, default=months)
df_filtered = df[df["month"].astype(str).isin(selected_months)].copy()

# -------------------------------
# Features & Target
# -------------------------------
X = df_filtered[selected_features]
y = df_filtered["revenue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Time-Series Cross Validation
# -------------------------------
tscv = TimeSeriesSplit(n_splits=5)
rmse_list = []
r2_list = []

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = Ridge(alpha=alpha_val)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_list.append(r2_score(y_test, y_pred))

# -------------------------------
# Metrics Cards
# -------------------------------
col1, col2 = st.columns(2)
col1.metric("Average RMSE", f"{np.mean(rmse_list):,.2f}")
col2.metric("Average R²", f"{np.mean(r2_list):.2f}")

# -------------------------------
# Feature Importance
# -------------------------------
coef_df = pd.DataFrame({
    "feature": selected_features,
    "coefficient": model.coef_
}).sort_values(by="coefficient", key=abs, ascending=False)

st.subheader("🔑 Feature Importance")
st.dataframe(coef_df.style.background_gradient(cmap="coolwarm"))
st.bar_chart(coef_df.set_index("feature")["coefficient"])

# -------------------------------
# Monthly Aggregated View
# -------------------------------
monthly_df = df_filtered.groupby("month").agg({
    "revenue":"sum",
    "social_spend":"sum",
    "google_spend":"sum"
}).reset_index()

st.subheader("📅 Monthly Revenue & Spend")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(monthly_df["month"].astype(str), monthly_df["revenue"], marker="o", label="Revenue", color="#1F618D")
ax.bar(monthly_df["month"].astype(str), monthly_df["social_spend"], alpha=0.6, label="Social Spend", color="#E67E22")
ax.bar(monthly_df["month"].astype(str), monthly_df["google_spend"], alpha=0.6, label="Google Spend", color="#27AE60")
ax.set_xlabel("Month")
ax.set_ylabel("Amount")
ax.set_title("Revenue & Marketing Spend per Month")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -------------------------------
# Predicted vs Actual Revenue
# -------------------------------
y_pred_full = model.predict(X_scaled)

st.subheader("📊 Predicted vs Actual Revenue")
fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(df_filtered["week"], y, label="Actual", color="#1F618D", linewidth=2)
ax2.plot(df_filtered["week"], y_pred_full, label="Predicted", color="#E74C3C", linewidth=2)
ax2.set_xlabel("Week")
ax2.set_ylabel("Revenue")
ax2.set_title("Actual vs Predicted Revenue Over Time")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# -------------------------------
# Residual Analysis
# -------------------------------
residuals = y - y_pred_full
st.subheader("Residual Analysis")
fig3, ax3 = plt.subplots(figsize=(12,4))
sns.histplot(residuals, kde=True, ax=ax3, color="#8E44AD")
ax3.set_title("Residuals Distribution")
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(12,4))
ax4.plot(df_filtered["week"], residuals, color="#2ECC71")
ax4.set_title("Residuals Over Time")
ax4.set_xlabel("Week")
ax4.set_ylabel("Residuals")
ax4.grid(True)
st.pyplot(fig4)

# -------------------------------
# Insights
# -------------------------------
st.subheader("💡 Insights")
if "average_price" in selected_features:
    price_coef = coef_df.loc[coef_df.feature=="average_price", "coefficient"].values[0]
    st.write(f"Price elasticity (linear approx): {price_coef:.4f}")
if "social_to_google" in selected_features:
    social_coef = coef_df.loc[coef_df.feature=="social_to_google", "coefficient"].values[0]
    st.write(f"Social -> Google -> Revenue mediated effect approx: {social_coef:.4f}")

# -------------------------------
# Interactive What-If Simulator
# -------------------------------
st.subheader("⚡ What-If Simulator")
st.markdown("Slide the controls below to simulate changes in promotions, emails, sms, and average price.")

col1, col2, col3, col4 = st.columns(4)
promo_slider = col1.slider("Promotions", int(df_filtered["promotions"].min()), int(df_filtered["promotions"].max()), int(df_filtered["promotions"].median()))
email_slider = col2.slider("Emails Sent", int(df_filtered["emails_send"].min()), int(df_filtered["emails_send"].max()), int(df_filtered["emails_send"].median()))
sms_slider = col3.slider("SMS Sent", int(df_filtered["sms_send"].min()), int(df_filtered["sms_send"].max()), int(df_filtered["sms_send"].median()))
price_slider = col4.slider("Average Price", float(df_filtered["average_price"].min()), float(df_filtered["average_price"].max()), float(df_filtered["average_price"].median()))

# Prepare simulation row
sim_data = {}
for feat in selected_features:
    if feat == "promotions":
        sim_data[feat] = promo_slider
    elif feat == "emails_send":
        sim_data[feat] = email_slider
    elif feat == "sms_send":
        sim_data[feat] = sms_slider
    elif feat == "average_price":
        sim_data[feat] = price_slider
    else:
        sim_data[feat] = df_filtered[feat].median()

sim_df = pd.DataFrame([sim_data])
sim_scaled = scaler.transform(sim_df)
sim_pred = model.predict(sim_scaled)[0]

st.success(f"💰 Predicted Revenue for selected inputs: {sim_pred:,.2f}")
