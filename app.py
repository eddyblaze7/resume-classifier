import streamlit as st
import joblib
import pypdf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Load model and vectorizer
model = joblib.load("model/resume_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# App configuration
st.set_page_config(page_title="Resume Classifier", page_icon="🧠")
st.title("🧠 Resume Classifier App")
st.markdown("💼 Analyze resumes using machine learning to identify likely job roles based on content. Powered by Scikit-learn and built with Streamlit.")
st.markdown("---")

# Sidebar branding and upload history toggle
st.sidebar.title("🧠 Resume Classifier")
st.sidebar.markdown("Built with Python, Scikit-learn, and Streamlit")
st.sidebar.markdown("---")
st.sidebar.subheader("🗂 Upload History")
if st.sidebar.checkbox("View Previous Uploads"):
    log_path = "logs/upload_history.csv"
    if os.path.exists(log_path):
        history_df = pd.read_csv(log_path)
        st.subheader("📁 Upload History")
        st.dataframe(history_df)
    else:
        st.info("No upload history found yet.")

# Extract text from PDF
def extract_text(file):
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Get top keywords influencing prediction
def get_top_keywords(vectorizer, model, class_label, top_n=10):
    class_index = list(model.classes_).index(class_label)
    coef = model.coef_[class_index]
    feature_names = vectorizer.get_feature_names_out()
    top_indices = np.argsort(coef)[-top_n:]
    return [feature_names[i] for i in reversed(top_indices)]

# Highlight keywords in HTML
def highlight_keywords(text, keywords):
    for kw in keywords:
        text = text.replace(kw, f'<span style="background-color: #ffe599; font-weight: bold">{kw}</span>')
    return text

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a Resume (PDF only)", type=["pdf"], key="resume_uploader")

if uploaded_file:
    resume_text = extract_text(uploaded_file)

    if resume_text:
        vectorized = vectorizer.transform([resume_text])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100

        top_keywords = get_top_keywords(vectorizer, model, prediction)

        # Display prediction and confidence in columns
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Predicted Role:\n**{prediction}**")
        with col2:
            st.info(f"📊 Confidence:\n**{confidence:.2f}%**")

        st.markdown("---")

        # Show all probabilities in a table
        proba_df = pd.DataFrame({
            'Role': model.classes_,
            'Confidence (%)': [round(p * 100, 2) for p in probabilities]
        }).sort_values(by='Confidence (%)', ascending=False)

        st.subheader("📊 All Role Predictions")
        st.table(proba_df.reset_index(drop=True))

        st.markdown("---")

        # Highlight top keywords in resume text
        highlighted_text = highlight_keywords(resume_text, top_keywords)

        st.subheader("📃 Resume Text with Highlighted Keywords")
        st.markdown(highlighted_text[:2000], unsafe_allow_html=True)  # Truncate to avoid overflow

        st.markdown("---")

        # Logging
        log_data = {
            "filename": uploaded_file.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

        os.makedirs("logs", exist_ok=True)
        log_path = "logs/upload_history.csv"
        log_exists = os.path.isfile(log_path)
        log_df = pd.DataFrame([log_data])
        log_df.to_csv(log_path, mode='a', header=not log_exists, index=False)

    else:
        st.warning("❌ Couldn't extract text from the uploaded PDF.")
