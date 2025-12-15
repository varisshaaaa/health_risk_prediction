import streamlit as st
import requests
import re
import pandas as pd
import os

# ---------------- CONFIG ----------------
API_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Health Advisor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Powered Health Advisory System")
st.markdown("### Intelligent, Multi-Level Risk Prediction & Advice")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio(
        "Select View",
        ["User Advisory", "Instructor / Admin Dashboard"]
    )
    st.divider()

    if mode == "User Advisory":
        st.header("Profile & Environment")
        city = st.text_input("City", "New York")
        age = st.number_input("Age", 0, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        st.info("Air Quality data is fetched automatically.")

# ---------------- ADMIN DASHBOARD ----------------
if mode == "Instructor / Admin Dashboard":
    st.header("🎓 Instructor Dashboard")

    st.subheader("Feature Store Logs")
    if st.button("Refresh Feature Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/features")
            if res.status_code == 200:
                st.dataframe(pd.DataFrame(res.json()))
            else:
                st.error("Failed to fetch feature logs.")
        except Exception as e:
            st.error(e)

    st.divider()

    st.subheader("New Symptom Learning Logs")
    if st.button("Refresh Symptom Logs"):
        try:
            res = requests.get(f"{API_URL}/logs/symptoms")
            if res.status_code == 200:
                st.dataframe(pd.DataFrame(res.json()))
            else:
                st.error("Failed to fetch symptom logs.")
        except Exception as e:
            st.error(e)

# ---------------- USER ADVISORY ----------------
else:
    st.subheader("Report Your Symptoms")

    # -------- FETCH SYMPTOMS FROM BACKEND --------
    try:
        res = requests.get(f"{API_URL}/symptoms")
        SYMPTOM_LIST = res.json() if res.status_code == 200 else []
    except:
        SYMPTOM_LIST = []

    # Regex for valid symptom names
    valid_pattern = re.compile(r"^[a-zA-Z][a-zA-Z\s_\-]*$")

    cols = st.columns(3)
    display_idx = 0

    symptoms_vector = []
    symptoms_selected = []

    st.caption("v2.5 – Smart UI & Advisory")

    # -------- MAIN SYMPTOM LOOP --------
    for i, symptom in enumerate(SYMPTOM_LIST):
        s = str(symptom).strip()
        s_lower = s.lower()

        # Blacklist junk values
        is_invalid = (
            not s or
            s_lower in {"nan", "none"} or
            "unnamed" in s_lower or
            not valid_pattern.match(s)
        )

        if is_invalid:
            # Keep feature slot but hide from UI
            symptoms_vector.append(0)
            continue

        col = cols[display_idx % 3]
        display_idx += 1

        display_name = s.replace("_", " ").title()
        checked = col.checkbox(
            display_name,
            key=f"sym_{i}_{s}"
        )

        symptoms_vector.append(1 if checked else 0)
        if checked:
            symptoms_selected.append(display_name)

    other_symptoms = st.text_input(
        "Other Symptoms (comma separated)",
        help="These help the model learn in future retraining"
    )

    # -------- PREDICT BUTTON --------
    if st.button("Analyze Health Risk", type="primary"):
        payload = {
            "age": age,
            "gender": gender,
            "city": city,
            "symptoms": symptoms_vector,
            "symptom_names": symptoms_selected,
            "other_symptoms": other_symptoms or ""
        }

        with st.spinner("Analyzing health, demographics, and air quality..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    st.divider()

                    r1, r2 = st.columns(2)

                    risk = result["risk_level"]
                    score = result["risk_score"]

                    color = (
                        "green" if risk == "LOW"
                        else "orange" if risk == "MEDIUM"
                        else "red"
                    )

                    with r1:
                        st.markdown(f"### Risk Level: :{color}[{risk}]")
                        st.progress(score)
                        st.caption(f"Score: {score:.2f}")

                    with r2:
                        aq = result["air_quality"]
                        st.metric("Air Quality (AQI)", aq["aqi"])
                        st.caption(
                            f"PM2.5: {aq['pollutants']['pm2_5']}"
                        )

                    st.divider()
                    st.subheader(f"Diagnosis: {result['disease']}")
                    st.success(result["advisory"])

                else:
                    st.error(response.text)

            except Exception as e:
                st.error(f"Backend connection failed: {e}")
                st.info("Ensure backend is running and reachable.")

st.markdown("---")
st.markdown(
    "*Disclaimer: This system provides AI-based guidance only. Always consult a medical professional.*"
)
