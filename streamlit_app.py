# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
from detector import FakeReviewDetector

# --- Gemini Setup ---
GEMINI_API_KEY = "AIzaSyBpB0CS_fQedYftEAEC6DixTYevfdKTJi0"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)

def ask_gemini_opinion(review_text):
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    prompt = (
        "You are an expert at detecting fake product reviews.\n"
        "Is the following review FAKE or REAL? Respond with FAKE or REAL and a short explanation.\n\n"
        f"Review:\n\"{review_text}\""
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# --- Streamlit Setup ---
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️")
st.title("🕵️ Fake Review Detector (SVM + Gemini)")
st.write("Enter a product review to detect whether it's likely **Fake** or **Real**.\nThe result is verified by both the trained model and Gemini AI.")

# Load model components
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
detector = joblib.load('feature_extractor.pkl')

# Input
user_input = st.text_area("✍️ Review Text", placeholder="Paste your review here...")

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        # Clean and prepare input
        cleaned_review = detector.clean_text(user_input)
        df = pd.DataFrame({'review': [cleaned_review]})
        traditional_features = detector.vectorizer.transform(df['review']).toarray()
        embeddings = detector.get_embeddings([user_input])
        X = np.hstack([traditional_features, embeddings])
        X_scaled = scaler.transform(X)

        # Predict with SVM
        pred = svm_model.predict(X_scaled)[0]
        prob = svm_model.predict_proba(X_scaled)[0][1]
        svm_label = "FAKE" if pred == 1 else "REAL"

        # Output SVM result
        st.subheader("🤖 SVM Model Prediction")
        if svm_label == "FAKE":
            st.error(f"🚨 This review is likely **FAKE** (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ This review is likely **REAL** (Confidence: {1 - prob:.2f})")

        # Gemini analysis
        st.subheader("🧠 Gemini AI Second Opinion")
        with st.spinner("Consulting Gemini..."):
            gemini_opinion = ask_gemini_opinion(user_input)
        st.info(gemini_opinion)

        # Final summary
        gemini_label = "FAKE" if "fake" in gemini_opinion.lower() else "REAL"
        st.subheader("📊 Final Verdict")

        if svm_label == gemini_label:
            if svm_label == "FAKE":
                st.error("❌ Both SVM and Gemini agree: FAKE")
            else:
                st.success("✅ Both SVM and Gemini agree: REAL")
        else:
            st.warning(f"⚠️ SVM and Gemini disagree:\n- SVM: **{svm_label}**\n- Gemini: **{gemini_label}**")
