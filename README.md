# Customer Analytics with Machine Learning

This project combines **Supervised** and **Unsupervised Machine Learning** techniques to analyze customer behavior. It includes:

* **Customer Churn Prediction** (Supervised Learning)
* **Customer Segmentation** (Unsupervised Learning)
* **Interactive Streamlit Dashboards** for both analyses

Both analyses are performed on real-world datasets and supported by extensive visualizations.

---

## 🚀 Live Dashboards

### Churn Prediction App
🔗 **Live Demo:** https://ml-customer-analysis-uadd9gemwynm44izglfeax.streamlit.app/

```bash
streamlit run churn_app.py
```
- Predict customer churn using 4 trained models (Logistic Regression, KNN, Decision Tree, SVM)
- Compare model performance with interactive charts
- Explore the dataset with visualizations

### Customer Segmentation App
🔗 **Live Demo:** https://ml-customer-analysis-80v7y9o5tgb.streamlit.app/

```bash
streamlit run segments_app.py
```
- View customer segments identified by K-Means clustering
- Analyze segment characteristics and distributions
- Explore cluster analysis visualizations

---

## 📌 Project Structure

```
ML-customer-analysis/
├── churn_app.py              # Streamlit app for churn prediction
├── segments_app.py           # Streamlit app for customer segmentation
├── supervised.ipynb          # Churn prediction (Supervised ML)
├── Unsupervised.ipynb        # Customer segmentation (Unsupervised ML)
├── models/                   # Trained models (.pkl files)
├── assets/                   # All generated plots and visuals
├── Datasets/                 # Telco + E-commerce datasets
└── requirements.txt          # Python dependencies
```

---

## 📊 1. Customer Churn Prediction (Supervised Learning)

Using the **Telco Customer Churn** dataset, the notebook performs end-to-end predictive modeling.

### 🔹 Steps Performed

* **Data Cleaning** (handling missing values in `TotalCharges`)
* **Label Encoding** (binary categorical features)
* **One‑Hot Encoding** (InternetService, Contract, PaymentMethod)
* **Feature Scaling** (StandardScaler)
* **Feature Selection** using **RFECV** → **18 optimal features**
* **Model Training & Evaluation** using:

  * Logistic Regression
  * K‑Nearest Neighbors (k=12)
  * Decision Tree (raw)
  * Decision Tree (pre‑pruned via GridSearchCV)
  * Support Vector Machine (Linear kernel)

### 📈 Model Performance

| Model                      | Accuracy   | F1 (Churn=1) |
| -------------------------- | ---------- | ------------ |
| **Logistic Regression**    | **0.8024** | **0.59**     |
| SVM (Linear)               | 0.7953     | 0.58         |
| Decision Tree (Pre‑Pruned) | 0.7853     | 0.57         |
| KNN (k=12)                 | 0.7844     | 0.54         |
| Decision Tree (No Pruning) | 0.7246     | 0.47         |

### 🔍 Visualizations

All located in `assets/`:

* Confusion Matrix Heatmap
* ROC Curves (all models)
* Precision–Recall Curves
* Random Forest Feature Importance
* SHAP Summary Plot

### 💡 Key Insights

* **Logistic Regression** performs best overall.
* Strong churn indicators include:

  * High MonthlyCharges
  * Short tenure
  * Month‑to‑month contracts
  * Electronic check payments
  * Lack of OnlineSecurity

---

## 🧩 2. Customer Segmentation (Unsupervised Learning)

Using an **e‑commerce transactions dataset**, the notebook builds meaningful customer segments.

### 🔹 Steps Performed

* Dropping missing values
* Feature engineering from `InvoiceDate`:

  * Hour, Day, Month, Weekday
* One‑Hot Encoding for `Country`
* Frequency encoding for StockCode & InvoiceNo
* Scaling with StandardScaler
* Dimensionality Reduction with PCA (95% variance → 22 components)

### 🧮 K‑Means Clustering

* **Elbow Method** → visual inertia drop
* **Silhouette Scores** → best at **K = 3**
* Clusters labeled as:

  * **Low Spenders**
  * **Regular Customers**
  * **High‑Value Customers**

### 📊 Visualizations

Found in `assets/`:

* Scree Plot (PCA variance)
* Elbow Plot
* Silhouette Score Bar Chart
* Cluster Size Distribution
* Bar Chart (Quantity vs Estimated Spend)
* PCA 2D Scatter Plot (PC1 vs PC2)
* PCA 3D Scatter Plot (PC1–PC3)

### 💡 Segment Insights

* **High‑Value customers** purchase larger quantities at higher price points.
* **Low Spenders** show minimal purchase volume.
* **Regular customers** fall in between with stable buying patterns.

---

## 🛠️ Technologies Used

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit
* Plotly
* Matplotlib / Seaborn
* SHAP
* Joblib

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ihabiba/ML-customer-analysis.git
cd ML-customer-analysis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the apps
streamlit run churn_app.py
streamlit run segments_app.py
```

---

## ✔️ Summary

This project applies machine learning to uncover **why customers churn** and **how customers group into behavioral segments**, using real datasets and strong model evaluation. It demonstrates end‑to‑end ML capability: preprocessing, feature selection, clustering, classification, and explainability.

---
