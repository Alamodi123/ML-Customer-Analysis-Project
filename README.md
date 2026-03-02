# 📊 Customer Analytics & Behavior Modeling

An end-to-end Machine Learning pipeline combining **Supervised** and **Unsupervised** techniques to decode customer behavior. This repository features predictive modeling for customer churn and clustering for customer segmentation, brought to life through interactive Streamlit dashboards.

**Core Focus Areas:**
* 📉 **Customer Churn Prediction** (Supervised Learning)
* 🧩 **Customer Segmentation** (Unsupervised Learning)

---

## 🌐 Live Interactive Dashboards

Experience the models in action before diving into the code:

* 🚀 **[Launch Churn Prediction App](https://ml-customer-analysis-1gsd7ol879b.streamlit.app/)** | Predict churn using four trained models (LR, KNN, DT, SVM) and explore dataset distributions.
* 🚀 **[Launch Customer Segmentation App](https://ml-customer-analysis-igluq4shomg.streamlit.app/)** | Analyze segment characteristics, cluster distributions, and 3D PCA visualizations.

---

## ⚙️ Quick Start & Installation

To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone [https://github.com/ihabiba/ML-customer-analysis.git](https://github.com/ihabiba/ML-customer-analysis.git)
cd ML-customer-analysis

# 2. Create and activate a virtual environment
python -m venv .venv
# Mac/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate  

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit applications
streamlit run churn_app.py
streamlit run segments_app.py

```

---

## 🔬 Core Analyses & Methodology

### Module 1: Churn Prediction (Supervised Learning)

**Dataset:** Telco Customer Churn

This module focuses on predicting which customers are likely to leave, utilizing an 18-feature optimal subset determined via **RFECV**.

**Pipeline:** Data Cleaning (handling `TotalCharges` NaNs) ➔ Label & One-Hot Encoding (`InternetService`, `Contract`, `PaymentMethod`) ➔ StandardScaler Feature Scaling ➔ Model Training.

**Performance Metrics:**

| Algorithm | Accuracy | F1-Score (Churn=1) |
| --- | --- | --- |
| 🏆 **Logistic Regression** | **0.8024** | **0.59** |
| Support Vector Machine (Linear) | 0.7953 | 0.58 |
| Decision Tree (Pre‑Pruned via GridSearch) | 0.7853 | 0.57 |
| K-Nearest Neighbors (k=12) | 0.7844 | 0.54 |
| Decision Tree (Raw / No Pruning) | 0.7246 | 0.47 |

> **💡 Key Insights:** Logistic Regression outperformed the complex models. The strongest indicators of customer churn are **high monthly charges, short tenure, month-to-month contracts, electronic check payments,** and a **lack of online security**.

**Generated Visuals (`/assets`):** Confusion Matrix Heatmap, ROC Curves, Precision–Recall Curves, Random Forest Feature Importance, SHAP Summary Plot.

### Module 2: Customer Segmentation (Unsupervised Learning)

**Dataset:** E-commerce Transactions

This module groups customers based on purchasing behavior to enable targeted marketing strategies.

**Pipeline:** Missing Value Removal ➔ Feature Engineering (`InvoiceDate`: Hour/Day/Month/Weekday) ➔ One-Hot Encoding (`Country`) ➔ Frequency Encoding (`StockCode`, `InvoiceNo`) ➔ StandardScaler ➔ **PCA Dimensionality Reduction** (95% variance retained across 22 components).

**Clustering Strategy (K-Means):**
Evaluated via the **Elbow Method** (visual inertia drop) and **Silhouette Scores**, identifying **K = 3** as the optimal number of clusters.

> **💡 Segment Insights:**
> * 🥇 **High‑Value Customers:** Purchase larger quantities at higher price points.
> * 🥈 **Regular Customers:** Exhibit stable buying patterns, falling squarely in the middle.
> * 🥉 **Low Spenders:** Show minimal purchase volume and frequency.
> 
> 

**Generated Visuals (`/assets`):** Scree Plot, Elbow Plot, Silhouette Score Bar Chart, Cluster Size Distribution, Quantity vs Estimated Spend Bar Chart, PCA Scatter Plots (2D & 3D).

---

## 📂 Repository Architecture

```text
ML-customer-analysis/
├── 📁 Datasets/          # Source data (Telco + E-commerce)
├── 📁 models/            # Serialized trained models (.pkl)
├── 📁 assets/            # Exported plots, graphs, and visual insights
├── 📓 supervised.ipynb   # Model training & evaluation (Churn)
├── 📓 Unsupervised.ipynb # PCA & Clustering (Segmentation)
├── 🐍 churn_app.py       # Streamlit UI for predictive modeling
├── 🐍 segments_app.py    # Streamlit UI for cluster analysis
└── 📄 requirements.txt   # Environment dependencies

```

---

## 🛠️ Technology Stack

**Languages & Core:** Python, Pandas, NumPy
**Machine Learning:** Scikit-learn, Joblib
**Visualization & Explainability:** Matplotlib, Seaborn, Plotly, SHAP
**Web Framework:** Streamlit

---

**Summary:** This repository demonstrates end‑to‑end ML capability—from raw data preprocessing and feature selection to model explainability and interactive deployment—answering *why* customers churn and *how* they behave.

```
