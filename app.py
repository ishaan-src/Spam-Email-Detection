import os
import sys
sys.path.append(os.path.abspath("src"))

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import clean_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

# ---------------- LOAD MODELS ----------------
models = joblib.load(os.path.join(MODELS_DIR, "models.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Spam Detection Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:800;
    color:#FFFFFF;
}
.card {
    background-color:#1e1e1e;
    padding:20px;
    border-radius:15px;
    margin-bottom:15px;
}
.metric {
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📧 Spam Email Detection System</div>', unsafe_allow_html=True)
st.write("Interactive ML dashboard with multiple classifiers")

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Spam Checker", "Model Evaluation"]
)

# =========================
# 1️⃣ SPAM CHECKER
# =========================
if menu == "Spam Checker":
    st.subheader("✉️ Enter Email Content")

    email_text = st.text_area("Paste the email text here", height=220)

    if st.button("Predict"):
        if email_text.strip() == "":
            st.warning("Please enter email text")
        else:
            clean_email = clean_text(email_text)
            X_input = vectorizer.transform([clean_email])

            st.subheader("📊 Model Predictions")

            for name, model in models.items():
                pred = model.predict(X_input)[0]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### {name}")

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_input)[0][1]
                    st.progress(int(prob * 100))
                    st.write(
                        "🚨 **Spam**" if pred == 1 else "✅ **Not Spam**",
                        f"(Confidence: {prob:.2f})"
                    )
                else:
                    st.write("🚨 **Spam**" if pred == 1 else "✅ **Not Spam**")

                st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 2️⃣ MODEL EVALUATION
# =========================
else:
    st.subheader("📈 Model Evaluation (Test Set Only)")

    df = pd.read_csv(DATA_PATH)
    X = df["text"].astype(str)
    y = df["spam"]

    # Proper train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_tfidf = vectorizer.transform(X_test)

    # ---------------- CONFUSION MATRICES ----------------
    st.subheader("🧮 Confusion Matrices")

    cols = st.columns(3)

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        cols[i % 3].pyplot(fig)

    # ---------------- ROC CURVES ----------------
    st.subheader("📊 ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        else:
            y_prob = model.predict(X_test_tfidf)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()

    st.pyplot(fig)
