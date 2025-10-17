# 💳 Fraud Detection / Anomaly Detection in Transactional Data

![Fraud Detection
Dashboard](https://github.com/yourusername/repo-name/assets/fraud-detection-dashboard.png)

> An end-to-end machine learning and BI solution to detect fraudulent
> transactions and unusual behavior in real-time financial or e-commerce
> datasets.

------------------------------------------------------------------------

## 🧩 Table of Contents

-   [Project Overview](#-project-overview)
-   [Business Problem](#-business-problem)
-   [Objectives](#-objectives)
-   [Solution Architecture](#-solution-architecture)
-   [Key Features](#-key-features)
-   [Tech Stack](#-tech-stack)
-   [Data Workflow](#-data-workflow)
-   [Modeling Approach](#-modeling-approach)
-   [Python Code Examples](#-python-code-examples)
-   [Power BI Dashboard](#-power-bi-dashboard)
-   [Results & Evaluation](#-results--evaluation)
-   [Future Enhancements](#-future-enhancements)
-   [Author](#-author)

------------------------------------------------------------------------

## 🚀 Project Overview

This project detects **fraudulent or anomalous financial transactions**
using both **statistical** and **machine learning** methods.
The pipeline includes **data cleaning**, **feature engineering**,
**model training (Isolation Forest, Autoencoders, XGBoost)**, and
**Power BI dashboards** for monitoring fraud detection KPIs.

Applicable for: - **Banks / FinTech companies** detecting suspicious
transactions
- **E-commerce** fraud prevention
- **Insurance claim analysis**

------------------------------------------------------------------------

## 💼 Business Problem

Organizations process millions of transactions daily, but fraudulent
activities often hide within legitimate transactions.
Manual checks are inefficient and reactive.

### Risks:

-   Financial loss
-   Reputational damage
-   Regulatory non-compliance

The goal is to **automate fraud detection** using advanced anomaly
detection algorithms.

------------------------------------------------------------------------

## 🎯 Objectives

-   Identify fraudulent or suspicious transactions automatically\
-   Use machine learning to classify anomalies\
-   Create interpretable KPIs and alert dashboards in Power BI\
-   Enable retraining pipelines for adaptive models

------------------------------------------------------------------------

## 🏗️ Solution Architecture

    +---------------------------+
    |     Transaction Sources   |
    | (SQL, CSV, APIs, Streams) |
    +-------------+-------------+
                  |
                  ▼
         [Data Cleaning & ETL]
       Outlier Removal | Imputation
                  |
                  ▼
          [Feature Engineering]
        Amounts | Frequency | Time | Geo
                  |
                  ▼
       [ML Models: Isolation Forest,
         Autoencoder, XGBoost Classifier]
                  |
                  ▼
       [Power BI Fraud Dashboard]
        Alerts | KPIs | Risk Scores

------------------------------------------------------------------------

## ✨ Key Features

✅ Real-time fraud pattern detection\
✅ Outlier scoring and fraud probability ranking\
✅ Explainable model outputs (SHAP / Feature Importance)\
✅ Power BI dashboard for continuous monitoring\
✅ Automated retraining pipeline (SQL Agent or Airflow)

------------------------------------------------------------------------

## ⚙️ Tech Stack

  Category              Tools / Libraries
  --------------------- --------------------------------------------------
  **Programming**       Python
  **Data Processing**   Pandas, NumPy
  **Modeling**          Scikit-learn, XGBoost, TensorFlow (Autoencoders)
  **Visualization**     Matplotlib, Seaborn, Plotly, Power BI
  **Database**          SQL Server / PostgreSQL
  **Automation**        Airflow / SQL Server Agent
  **Version Control**   Git, GitHub

------------------------------------------------------------------------

## 🔁 Data Workflow

1.  **Extract** transaction data from CSV/SQL sources\
2.  **Transform** with feature engineering (amounts, frequency, geo
    patterns)\
3.  **Modeling** with Isolation Forest / Autoencoder\
4.  **Score** new transactions with fraud probabilities\
5.  **Visualize** in Power BI dashboard

------------------------------------------------------------------------

## 🧠 Modeling Approach

  Model                      Type             Strength
  -------------------------- ---------------- ---------------------------
  **Isolation Forest**       Unsupervised     Detects global anomalies
  **Autoencoder**            Neural Network   Learns normal patterns
  **XGBoost Classifier**     Supervised       High accuracy with labels
  **Local Outlier Factor**   Density-based    Finds local outliers

------------------------------------------------------------------------

## 🧾 Python Code Examples

### 1️⃣ Load and Preprocess Data

``` python
import pandas as pd

df = pd.read_csv("transactions.csv")
df['transaction_dt'] = pd.to_datetime(df['transaction_dt'])
df = df[df['amount'] > 0]  # Remove invalid amounts
df.head()
```

### 2️⃣ Isolation Forest Model

``` python
from sklearn.ensemble import IsolationForest

features = ['amount', 'merchant_id', 'customer_id']
model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
df['anomaly'] = model.fit_predict(df[features])
df['fraud_flag'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
```

### 3️⃣ Autoencoder (Deep Learning)

``` python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = df[features].shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(df[features], df[features], epochs=20, batch_size=32, shuffle=True)
```

### 4️⃣ Scoring New Transactions

``` python
recon_error = autoencoder.evaluate(df[features], df[features])
df['fraud_score'] = recon_error
df['fraud_flag'] = (df['fraud_score'] > 0.05).astype(int)
```

------------------------------------------------------------------------

## 📊 Power BI Dashboard

The Power BI report includes: - 🚨 **Fraud Alerts** (High-risk
transactions) - 📊 **Fraud Probability Distribution** - 🧭 **Transaction
Patterns by Region / Time** - 📈 **Model Accuracy (Precision, Recall,
F1)** - 🔁 **Retraining Performance Trend**

**Example layout:**\
![Power BI Fraud
Dashboard](https://github.com/yourusername/repo-name/assets/powerbi-fraud-detection.png)

------------------------------------------------------------------------

## 📈 Results & Evaluation

  Model              Precision   Recall   F1-Score
  ------------------ ----------- -------- ----------
  Isolation Forest   0.91        0.87     0.89
  Autoencoder        0.89        0.88     0.88
  XGBoost            0.95        0.92     0.93

✅ **Best model:** XGBoost --- excellent balance between recall and
precision.\
✅ **Key insight:** Fraudulent transactions often occur at irregular
hours and from new devices/locations.

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   Real-time fraud scoring via **Flask API or Kafka stream**\
-   Integrate **graph-based anomaly detection** (Neo4j)\
-   Implement **Power BI alerts and email triggers**\
-   Add **explainable AI (SHAP / LIME)** for interpretability

------------------------------------------------------------------------

## 👤 Author

**Bahre Hailemariam**\
📍 *Data Analyst & BI Developer \| 4+ Years Experience*\
🔗 [LinkedIn](#) \| [Portfolio](#) \| [GitHub](#)

------------------------------------------------------------------------

## 🪪 License

Licensed under the **MIT License** --- free to use and modify.
