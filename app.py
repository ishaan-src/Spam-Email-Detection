import os
import sys
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("src"))
from data_preprocessing import clean_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ---------------- SETTINGS ----------------
st.set_page_config(page_title="Spam Detection", layout="wide", page_icon="🛡️")

# ---------------- STYLING ----------------
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background-color: #0e1117;
    }
    /* Custom Card Design */
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #31333f;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 5px solid #ff4b4b;
        background-color: #262730;
    }
    .spam-label { color: #ff4b4b; font-weight: bold; font-size: 20px; }
    .ham-label { color: #00c0f2; font-weight: bold; font-size: 20px; }
    
    /* Header styling */
    .main-title {
    font-size: 70px !important;
    font-weight: 800 !important;
    color: white !important;
    text-align: left;
    margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- PATHS & LOADING ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

df = pd.read_csv(DATA_PATH)

@st.cache_resource
def load_assets():
    models = joblib.load(os.path.join(MODELS_DIR, "models.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    return models, vectorizer

models, vectorizer = load_assets()

# ---------------- HEADER ----------------
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown(
    '<div class="main-title">Email Spam Detection</div>',
    unsafe_allow_html=True
    )
  

st.divider()

# ---------------- SIDEBAR NAVIGATION ----------------
page = st.sidebar.radio(
    "Navigation",
    ["Live Scanner", "Analytics", "Dataset Info"],
    index=0
)


# =========================
# TAB 1: LIVE SCANNER
# =========================
if page == "Live Scanner":
    st.header("Live Scanner")
    col1, col2 = st.columns([2, 1])

    with col1:
        # st.subheader("Analyze Message")
        email_text = st.text_area("Paste the email content below:", placeholder="Enter text here...", height=250)
        predict_btn = st.button("Scan", use_container_width=True, type="primary")

    with col2:
        st.subheader("Result Summary")
        if predict_btn and email_text.strip():
            clean_email = clean_text(email_text)
            X_input = vectorizer.transform([clean_email])
            
            # Aggregating results for a top-level summary
            all_preds = [m.predict(X_input)[0] for m in models.values()]
            avg_spam_score = sum(all_preds) / len(all_preds)
            
            if avg_spam_score > 0.5:
                st.error("### 🚨 HIGH RISK: SPAM")
                st.write("Most models flagged this content as suspicious.")
            else:
                st.success("### ✅ LOW RISK: HAM")
                st.write("This message appears safe.")
        else:
            st.info("Awaiting input for analysis...")

    if predict_btn and email_text.strip():
        st.divider()
        st.subheader("Detailed Model Breakdown")
        
        # Grid of model results
        m_cols = st.columns(len(models))
        for i, (name, model) in enumerate(models.items()):
            with m_cols[i]:
                pred = model.predict(X_input)[0]
                label = "SPAM" if pred == 1 else "CLEAN"
                color = "red" if pred == 1 else "green"
                
                st.markdown(f"""
                <div style="background:#1e2130; padding:15px; border-radius:10px; border-top: 4px solid {color};">
                    <small>{name}</small><br>
                    <b style="color:{color}; font-size:20px;">{label}</b>
                </div>
                """, unsafe_allow_html=True)
                
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_input)[0][1]
                    st.progress(float(prob))
                    st.caption(f"Confidence: {prob:.2%}")

# =========================
# TAB 2: ANALYTICS
# =========================
# st.divider()
elif page == "Analytics":
    st.header("Analytics")
    st.subheader("Model Performance Comparison")

    # Load and prep data only when needed
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str), df["spam"], test_size=0.2, random_state=42, stratify=df["spam"]
    )
    X_test_tfidf = vectorizer.transform(X_test)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🧮 Confusion Matrices")
        sub_tab_names = list(models.keys())
        matrix_tabs = st.tabs(sub_tab_names)
        
        for i, (name, model) in enumerate(models.items()):
            with matrix_tabs[i]:
                y_pred = model.predict(X_test_tfidf)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

    with c2:
        st.markdown("#### 📈 ROC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        for name, model in models.items():
            y_prob = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test_tfidf)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        
        ax_roc.plot([0, 1], [0, 1], "w--", alpha=0.5)
        ax_roc.set_facecolor('#0e1117')
        fig_roc.patch.set_facecolor('#0e1117')
        ax_roc.tick_params(colors='white')
        ax_roc.legend()
        st.pyplot(fig_roc)

# =========================
# TAB 3: DATASET INFO
# =========================

elif page == "Dataset Info":
    st.header("Dataset Info")
    st.subheader("Data Overview")
    st.dataframe(df.head(10), use_container_width=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("Class Distribution")
        dist = df['spam'].value_counts()
        st.bar_chart(dist)
    with col_d2:
        st.write("Dataset Statistics")
        st.write(df.describe())

