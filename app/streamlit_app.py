"""
app/streamlit_app.py - Web UI for heart disease prediction.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from src.predict import predict, load_model_meta

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Predictor")
st.caption("Enter patient details to get a prediction.")
st.divider()

# Sidebar - model info
with st.sidebar:
    st.header("Model Info")
    meta = load_model_meta()
    st.write(f"**Model:** {meta.get('best_model', 'N/A')}")
    st.write(f"**Accuracy:** {meta.get('accuracy', 'N/A')}")
    st.write(f"**AUC-ROC:** {meta.get('auc', 'N/A')}")

# Input form
col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"{x} - {['Typical','Atypical','Non-anginal','Asymptomatic'][x]}")
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=130)
    chol     = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")
    restecg  = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: f"{x} - {['Normal','ST-T Abnormal','LV Hypertrophy'][x]}")

with col2:
    thalach  = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=165)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
    oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    slope    = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: f"{x} - {['Upsloping','Flat','Downsloping'][x]}")
    ca       = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: f"{x} - {['Normal','Fixed Defect','Reversible Defect','Unknown'][x]}")

st.divider()

if st.button("🔍 Predict", type="primary", use_container_width=True):
    patient = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                   fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                   oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
    try:
        result = predict(patient)

        # Result display
        if result["prediction"] == 1:
            st.error(f"❌ {result['label']}")
        else:
            st.success(f"✅ {result['label']}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probability",  f"{result['probability']*100:.1f}%")
        col_b.metric("Risk Level",   result["risk_level"])
        col_c.metric("Model",        result["model_used"])

        st.progress(result["probability"])

        with st.expander("Input Summary"):
            st.dataframe(pd.DataFrame([patient]).T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(f"Error: {e}")

st.caption("UCI Cleveland Heart Disease Dataset | Python 3.14.7 + Streamlit")