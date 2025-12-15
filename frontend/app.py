import streamlit as st
import requests
import pandas as pd

# Backend URL
API_URL = "http://localhost:8000"
# For Railway deployment, this might need to change, but usually handled via env or same-network if using internal URLs in Docker.
# Since Streamlit runs client-side/browser access, it uses public URL. 
# We will default to localhost but check env.
import os
API_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Health Advisor", page_icon="🩺", layout="wide")

st.title("🩺 AI-Powered Health Advisory System")
st.markdown("### Intelligent, Multi-Level Risk Prediction & Advice")

# --- Sidebar: User Info ---
with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio("Select View", ["User Advisory", "Instructor / Admin Dashboard"])
    st.divider()
    
    if mode == "User Advisory":
        st.header("Profile & Environment")
        city = st.text_input("City", "New York")
        age = st.number_input("Age", 0, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        st.info("Air Quality data will be fetched automatically.")

# --- Main Logic ---
if mode == "Instructor / Admin Dashboard":
    st.header("🎓 Instructor Dashboard - Feature Store & Learning Logs")
    
    st.subheader("Feature Store (Prediction Logs)")
    st.markdown("Raw data of inputs (features) and model outputs (predictions).")
    if st.button("Refresh Feature Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/features")
            if res.status_code == 200:
                df = pd.DataFrame(res.json())
                st.dataframe(df)
            else:
                st.error("Failed to fetch logs.")
        except Exception as e:
            st.error(f"Error: {e}")
            
    st.divider()
    
    st.subheader("Dynamic Learning Logs (New Symptoms)")
    st.markdown("Symptoms reported by users that are not in the training set.")
    if st.button("Refresh Symptom Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/symptoms")
            if res.status_code == 200:
                df = pd.DataFrame(res.json())
                st.dataframe(df)
            else:
                st.error("Failed to fetch logs.")
        except Exception as e:
            st.error(f"Error: {e}")

else:
    # --- User Advisory View ---
    st.subheader("Report Your Symptoms")

    # Load symptom list (Hardcoded for demo, normally fetched)
    # Matching the CSV columns order is critical for the vector
    SYMPTOM_LIST = [
        "fever", "cough", "fatigue", "headache", "nausea", 
        "skin_rash", "joint_pain"
    ]
    # Note: This must match the training data columns (minus 'disease') in order.

    cols = st.columns(3)
    symptoms_selected = []
    symptoms_vector = []

    for i, symptom in enumerate(SYMPTOM_LIST):
        col = cols[i % 3]
        checked = col.checkbox(symptom.capitalize())
        symptoms_selected.append(symptom) if checked else None
        symptoms_vector.append(1 if checked else 0)

    other_symptoms = st.text_input("Other Symptoms (comma separated)", help="For future learning")

    if st.button("Analyze Health Risk", type="primary"):
        if not any(symptoms_vector):
            st.warning("Please select at least one symptom.")
        else:
            # Prepare payload
            payload = {
                "age": age,
                "gender": gender,
                "city": city,
                "symptoms": symptoms_vector,
                "symptom_names": symptoms_selected,
                "other_symptoms": other_symptoms if other_symptoms else None
            }
            
            with st.spinner("Analyzing symptoms, demographics, and air quality..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display Results
                        st.divider()
                        
                        # Columns for Risk and AQI
                        r_col1, r_col2 = st.columns(2)
                        
                        risk_level = result['risk_level']
                        color = "green" if risk_level == "LOW" else "orange" if risk_level == "MEDIUM" else "red"
                        
                        with r_col1:
                            st.markdown(f"### Risk Level: :{color}[{risk_level}]")
                            st.progress(result['risk_score'])
                            st.caption(f"Score: {result['risk_score']:.2f}")

                        with r_col2:
                            aq = result['air_quality']
                            st.metric("Air Quality (AQI)", aq['aqi'], delta_color="inverse")
                            st.caption(f"Pollutants: PM2.5: {aq['pollutants']['pm2_5']}")

                        st.divider()
                        st.subheader(f"Diagnosis: {result['disease']}")
                        
                        # Human-like Advisory
                        st.markdown("### 📋 AI Advisory")
                        st.success(result['advisory'])
                        
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
                    st.info("Ensure the backend is running on port 8000.")

st.markdown("---")
st.markdown("*Disclaimer: This is an AI advisory system. Always consult a real doctor.*")
