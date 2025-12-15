import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/disease_model.pkl"
COLUMNS_PATH = "models/symptom_columns.pkl"
SYMPTOM_REGISTRY_PATH = "data/symptoms_master.json"
PRECAUTIONS_PATH = "data/clean_precautions.csv"

# -----------------------------
# Load model & columns
# -----------------------------
model = joblib.load(MODEL_PATH)
symptom_columns = joblib.load(COLUMNS_PATH)

# -----------------------------
# Load symptom registry
# -----------------------------
with open(SYMPTOM_REGISTRY_PATH) as f:
    symptom_registry = json.load(f)

# -----------------------------
# Load disease precautions
# -----------------------------
df_prec = pd.read_csv(PRECAUTIONS_PATH)
disease_precautions = dict(zip(df_prec["disease"], df_prec["precautions"]))

# -----------------------------
# Risk-based precautions
# -----------------------------
RISK_PRECAUTIONS = {
    "LOW": [
        "Monitor symptoms for 24–48 hours",
        "Maintain hydration",
        "Ensure adequate rest"
    ],
    "MEDIUM": [
        "Consult a healthcare professional",
        "Avoid strenuous activity",
        "Follow prescribed medication if any"
    ],
    "HIGH": [
        "Seek immediate medical attention",
        "Do not self-medicate",
        "Emergency care may be required"
    ]
}

# -----------------------------
# Helper functions
# -----------------------------
def calculate_risk(confidence):
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def generate_reason(user_vector, symptom_columns):
    return [symptom_columns[i].replace("_", " ") for i, v in enumerate(user_vector) if v == 1]

def generate_precautions(disease, risk):
    precautions = []
    if disease in disease_precautions:
        precautions.extend(disease_precautions[disease].split(","))
    precautions.extend(RISK_PRECAUTIONS[risk])
    return list(set([p.strip() for p in precautions]))

def update_symptom_registry(new_symptom):
    pending = symptom_registry["pending_symptoms"]
    if new_symptom in pending:
        pending[new_symptom] += 1
    else:
        pending[new_symptom] = 1

    if pending[new_symptom] >= symptom_registry["confirmation_threshold"]:
        symptom_registry["confirmed_symptoms"].append(new_symptom)
        del pending[new_symptom]

    # Save updated registry
    with open(SYMPTOM_REGISTRY_PATH, "w") as f:
        json.dump(symptom_registry, f, indent=2)

def predict_disease(user_vector):
    probs = model.predict_proba([user_vector])[0]
    idx = np.argmax(probs)
    return model.classes_[idx], probs[idx]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Symptom-Based Disease Predictor")

st.subheader("Select your symptoms:")

# User selects known symptoms
user_selected = []
for symptom in symptom_registry["confirmed_symptoms"]:
    if st.checkbox(symptom.replace("_", " ")):
        user_selected.append(symptom)

# User can input new symptom
new_symptom = st.text_input("Other symptom (if any):").strip().lower()

if st.button("Predict Disease"):
    # Prepare user vector
    user_vector = [1 if col in user_selected else 0 for col in symptom_columns]

    # Handle new symptom
    if new_symptom:
        update_symptom_registry(new_symptom)
        st.info(f"✅ New symptom '{new_symptom}' recorded and will be added after confirmation.")

    # Prediction
    disease, confidence = predict_disease(user_vector)
    risk = calculate_risk(confidence)
    reason = generate_reason(user_vector, symptom_columns)
    precautions = generate_precautions(disease, risk)

    # Display results
    st.success(f"Predicted Disease: {disease}")
    st.write(f"Confidence: {confidence:.2f}")
    st.warning(f"Risk Level: {risk}")
    st.write("**Reason based on symptoms:**")
    st.write(", ".join(reason))
    st.write("**Precautions:**")
    for p in precautions:
        st.write(f"- {p}")
