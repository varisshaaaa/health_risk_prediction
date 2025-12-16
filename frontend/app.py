import streamlit as st
import requests
import re
import pandas as pd
import os
import plotly.express as px

# ---------------- CONFIG ----------------
API_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Health Advisor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Powered Health Advisory System v3.0")
st.caption("Integrated Risk Assessment (Symptoms, Demographics, Environment)")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["User Advisory", "Instructor Dashboard", "Model Reports"])

# ==========================================
# TAB 1: USER ADVISORY
# ==========================================
with tab1:
    st.header("Assess Your Health Risk")
    
    with st.expander("📝 Patient Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.text_input("City", "New York")
        with col2:
            age = st.number_input("Age", 0, 120, 30)
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        st.info(f"📍 Analysing environmental factors for **{city}**...")

    st.subheader("Report Symptoms")
    
    # -------- FETCH SYMPTOMS --------
    try:
        res = requests.get(f"{API_URL}/symptoms", timeout=5)
        SYMPTOM_LIST = res.json() if res.status_code == 200 else []
    except:
        SYMPTOM_LIST = []
        st.error("Backend not reachable. Ensure API is running.")

    # Regex for valid symptom names
    valid_pattern = re.compile(r"^[a-zA-Z][a-zA-Z\s_\-]*$")
    
    symptoms_vector = []
    symptoms_selected = []
    
    # Grid Layout for Symptoms
    with st.container():
        cols = st.columns(4)
        display_idx = 0
        
        for i, symptom in enumerate(SYMPTOM_LIST):
            s = str(symptom).strip()
            if not s or s.lower() in {"nan", "none"} or not valid_pattern.match(s):
                symptoms_vector.append(0)
                continue
                
            col = cols[display_idx % 4]
            display_idx += 1
            
            display_name = s.replace("_", " ").title()
            if col.checkbox(display_name, key=f"sym_{i}"):
                symptoms_vector.append(1)
                symptoms_selected.append(display_name)
            else:
                symptoms_vector.append(0)

    other_symptoms = st.text_input("Other Symptoms (comma separated)", placeholder="e.g. dizziness, anxiety")
    
    if st.button("🚀 Analyze Health Risk", type="primary"):
        payload = {
            "age": age,
            "gender": gender,
            "city": city,
            "symptoms": symptoms_vector,
            "symptom_names": symptoms_selected,
            "other_symptoms": other_symptoms or ""
        }
        
        with st.spinner("Calculating weighted risk scores..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # --- RESULTS SECTION ---
                    st.divider()
                    
                    # Top Level Metrics
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        risk = result["risk_level"]
                        score = result["risk_score"]
                        color = "green" if risk == "LOW" else "orange" if risk == "MODERATE" else "red"
                        if risk == "CRITICAL": color = "red"
                        
                        st.metric("Risk Level", risk, delta=f"{score:.1f}%")
                        st.progress(score / 100)
                    
                    with r2:
                        st.metric("Predicted Condition", result["disease"])
                    
                    with r3:
                        aq = result["air_quality"]
                        st.metric("Air Quality (AQI)", f"{aq['aqi']}/5", delta_color="inverse")
                    
                    # --- BREAKDOWN CHART (50/30/20) ---
                    st.subheader("Risk Contribution Breakdown")
                    comps = result.get("components", {})
                    if comps:
                        chart_data = pd.DataFrame({
                            "Factor": ["Symptoms (50%)", "Demographics (30%)", "Environment (20%)"],
                            "Contribution": [
                                comps.get("symptom_contribution", 0),
                                comps.get("demographic_contribution", 0),
                                comps.get("environmental_contribution", 0)
                            ]
                        })
                        fig = px.bar(chart_data, x="Contribution", y="Factor", orientation='h', text_auto='.1f')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # --- ADVISORY & PRECAUTIONS ---
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"**AI Advisory:**\n\n{result['advisory']}")
                    
                    with c2:
                        st.warning("**Recommended Precautions:**")
                        for p in result.get("precautions", []):
                            st.markdown(f"- {p}")

                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# ==========================================
# TAB 2: INSTRUCTOR DASHBOARD
# ==========================================
with tab2:
    st.header("🎓 Instructor Control Panel")
    
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    col1.metric("Backend Status", "Online" if requests.get(API_URL).status_code==200 else "Offline")
    
    if st.button("Trigger Manual Retraining"):
        try:
            res = requests.post(f"{API_URL}/admin/retrain")
            if res.status_code == 200:
                st.success("Retraining Job Triggered!")
            else:
                st.error("Failed to trigger.")
        except:
            st.error("Connection Error")

# ==========================================
# TAB 3: MODEL REPORTS
# ==========================================
with tab3:
    st.header("📊 Model Performance & Logs")
    
    st.subheader("Feature Store (Recent Predictions)")
    if st.button("Refresh Feature Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/features")
            if res.status_code == 200:
                df = pd.DataFrame(res.json())
                st.dataframe(df)
            else:
                st.error("Failed to fetch logs.")
        except Exception as e:
            st.error(e)

    st.subheader("New Symptom Discovery")
    if st.button("Refresh Symptom Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/symptoms")
            if res.status_code == 200:
                df = pd.DataFrame(res.json())
                st.dataframe(df)
            else:
                st.error("Failed to fetch logs.")
        except Exception as e:
            st.error(e)

    # Placeholder for Model Version History if implemented
    st.subheader("Model Configuration")
    st.json({
        "Algorithm": "CatBoost Classifier",
        "Risk Weights": {"Symptoms": 0.5, "Demographics": 0.3, "Environment": 0.2},
        "Retraining Interval": "5 Hours"
    })
