📊 Marketing Mix Modeling (MMM) Dashboard

A Streamlit-based interactive dashboard for Marketing Mix Modeling (MMM) with a Google-as-mediator assumption. This project analyzes paid media, direct response channels, pricing, promotions, and social metrics to model and predict revenue over time.

🔹 Features

Data Visualization

Weekly revenue trends

Predicted vs. actual revenue

Residual analysis

Modeling

Ridge Regression with time-series cross-validation

Feature scaling & lag features

Causal mediation modeling: Google spend as a mediator between social media spend and revenue

Interactive Simulation

Adjust promotions, emails, SMS, and average price

Real-time revenue predictions

Feature Importance

Visual ranking of drivers of revenue

Insights for growth and marketing decisions

Aesthetic Dashboard

Custom colors, gradients, and interactive plots using Streamlit, Matplotlib, and Seaborn

🔹 Installation

Clone the repository:

git clone https://github.com/mehga09/MMM-Modeling-with-Mediation-Assumption.git
cd MMM-Modeling-with-Mediation-Assumption


Install required packages:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run mmm_dashboard_v3.py

🔹 Dataset

The dashboard uses marketing_dataset.csv, which contains:

Paid media spend: Facebook, TikTok, Instagram, Snapchat, Google

Direct response levers: Emails, SMS

Pricing & promotions

Social followers

Revenue

Columns Example:

week, facebook_spend, google_spend, tiktok_spend, instagram_spend,
snapchat_spend, social_followers, average_price, promotions, emails_send,
sms_send, revenue

🔹 How It Works

Preprocesses the data:

Fills missing values

Creates lag features for Google spend

Generates total social spend and interaction features

Extracts week-of-year for seasonality

Builds a Ridge Regression model:

Uses selected features

Scales features

Performs time-series cross-validation for robust evaluation

Shows interactive results:

Feature importance

Predicted vs actual revenue

Residual analysis

“What-If” scenario simulation for decision-making

🔹 Insights & Recommendations

Understand price elasticity and impact of promotions

Analyze mediated effects of social spend on Google spend and revenue

Identify high-impact marketing channels

Use simulation to test different marketing strategies before execution

🔹 Tech Stack

Python 3.x

Streamlit

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

🔹 Authors

MEHGA Rani
GitHub: mehga09

🔹 License

This project is open-source and free to use for educational purposes.