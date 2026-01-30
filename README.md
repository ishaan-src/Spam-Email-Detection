# 📧 Spam Email Detection System

A full-stack Machine Learning web application that classifies emails as **Spam** or **Not Spam** using multiple classical ML models.  
The system provides **real-time predictions**, **confidence visualization**, and **comprehensive model evaluation dashboards**, and is deployed using **Streamlit**.

---

## 🚀 Features

### 🔍 Spam Checker
- Enter any email text and get predictions from:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- Visual confidence bars for probabilistic models
- Side-by-side comparison of model outputs

### 📊 Model Evaluation Dashboard
- Confusion matrices for all models (single-page view)
- ROC curve comparison (test set only)
- Clean and interactive visualizations
- Proper train–test separation (no data leakage)

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing**
   - Lowercasing
   - Special character removal
   - Stopword removal using NLTK

2. **Feature Extraction**
   - TF-IDF Vectorization
   - Unigrams and bigrams
   - Fixed feature space for fair model comparison

3. **Models Implemented**
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
   - Random Forest
   - XGBoost

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - Confusion Matrix

---

## 🛠️ Tech Stack

- **Language:** Python  
- **ML Libraries:** Scikit-learn, XGBoost, NLTK  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Web Framework:** Streamlit  
- **Model Persistence:** Joblib  

---



