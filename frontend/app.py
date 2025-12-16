import streamlit as st
import requests
import re
import pandas as pd
import os
import plotly.express as px

# ---------------- CONFIG ----------------
API_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Health Advisor V4.0",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Powered Health Advisory System V4.0")
st.caption("Integrated Risk Assessment • Smart Symptom Learning • Dynamic Web Scraping")

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
    
    symptoms_selected = []
    
    # Grid Layout for Symptoms
    with st.container():
        cols = st.columns(4)
        display_idx = 0
        
        # Limit display for UI performance, use text input for rest
        # Assuming typical top 50 common symptoms
        display_limit = 40 
        
        for i, symptom in enumerate(SYMPTOM_LIST[:display_limit]):
            s = str(symptom).strip()
            if not s or s.lower() in {"nan", "none"} or not valid_pattern.match(s):
                continue
                
            col = cols[display_idx % 4]
            display_idx += 1
            
            display_name = s.replace("_", " ").title()
            if col.checkbox(display_name, key=f"sym_{i}"):
                symptoms_selected.append(s)

    st.markdown("---")
    st.markdown("**Describe Additional Symptoms (Natural Language)**")
    other_symptoms = st.text_area("Example: 'I have a splitting headache and feeling nausea'", height=80)
    
    if st.button("🚀 Analyze Health Risk", type="primary"):
        # Combine inputs
        # For the new API, we can just pass the list of strings + the raw text
        # and let the backend 'symptom_manager' merge them.
        
        payload = {
            "age": age,
            "gender": gender,
            "city": city,
            "symptoms": symptoms_selected, # List of strings from checkboxes
            "other_symptoms": other_symptoms # Raw text
        }
        
        with st.spinner("AI is analyzing symptoms, checking databases, and verifying air quality..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # --- RESULTS SECTION ---
                    st.divider()
                    
                    # Top Level Metrics
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        # Use overall health risk score
                        score = result["overall_health_risk"]
                        severity = result["disease_severity"]
                        
                        color = "green" if score < 40 else "orange" if score < 70 else "red"
                        
                        st.metric("Health Risk Score", f"{score:.1f}/100", delta=severity, delta_color="inverse")
                        st.progress(score / 100)
                    
                    with r2:
                        st.metric("Predicted Condition", result["disease"])
                        st.caption(f"Probability: {result.get('probability', 0):.1f}%")
                    
                    with r3:
                        aq = result["air_quality"]
                        st.metric("Air Quality", f"{aq.get('status', 'N/A')}", help=f"AQI: {aq.get('aqi', 0)}")
                    
                    # --- SMART LEARNING ALERTS ---
                    new_syms = result.get("new_symptoms_detected", [])
                    if new_syms:
                        st.toast(f"New Symptoms Detected: {len(new_syms)}", icon="🧠")
                        st.success(f"**🧠 Smart Learning Active**: The system encountered new symptoms `{new_syms}`. \n\n"
                                   f"Initiated web scraping protocol to learn about associated diseases. Database updating in background...")

                    # --- MATCHED SYMPTOMS (Hidden as per request) ---
                    # with st.expander("🔍 Symptom Analysis"):
                    #    st.write("Using Fuzzy Matching & NLP, we identified:")
                    #    st.write(result.get("matched_symptoms", []))
                    
                    # --- ADVISORY & PRECAUTIONS ---
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(result['advisory'])
                    
                    with c2:
                        st.warning(f"**🛡️ Recommended Precautions for {result['disease']}**")
                        precautions = result.get("precautions", [])
                        if precautions and precautions != ["Not available"]:
                             for p in precautions:
                                st.markdown(f"- {p}")
                        else:
                            st.info("System is currently scraping specific precautions for this condition. Please try again in 1 minute.")
 
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                import traceback
                st.error(f"Connection Failed: {e}")
                st.code(traceback.format_exc())

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
    st.header("📊 Model Metrics")
    
    # FETCH PERFORMANCE
    try:
        res = requests.get(f"{API_URL}/logs/performance")
        if res.status_code == 200:
            hist_data = res.json()
            if hist_data:
                df_hist = pd.DataFrame(hist_data)
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                
                st.subheader("📈 Model Accuracy Over Time")
                fig = px.line(df_hist, x='timestamp', y='accuracy', markers=True, title="Retraining Performance History")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_hist.sort_values('timestamp', ascending=False))
            else:
                st.info("No training history available yet.")
        else:
            st.error("Failed to fetch performance logs.")
            
    except Exception as e:
        st.error(f"Error loading graphs: {e}")

    st.divider()
    
    st.subheader("Recent Feature Logs")
    if st.button("Refresh Prediction Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/features")
            if res.status_code == 200:
                st.dataframe(pd.DataFrame(res.json()))
        except Exception as e:
            st.error(e)
