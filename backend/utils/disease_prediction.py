import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "symptoms_and_disease.csv")
PRECAUTIONS_PATH = os.path.join(PROJECT_ROOT, "backend", "precautions.csv")

class DiseaseRiskOrchestrator:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.df = None
        self.load_resources()

    def load_resources(self):
        """Loads model, encoder, and dataset."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
            
            if os.path.exists(DATA_PATH):
                self.df = pd.read_csv(DATA_PATH)
                
            self.precautions_df = pd.read_csv(PRECAUTIONS_PATH) if os.path.exists(PRECAUTIONS_PATH) else pd.DataFrame()
            
        except Exception as e:
            print(f"Error loading resources: {e}")

    def predict_diseases(self, symptoms_input, top_n=5):
        """
        User's specific logic for prediction and severity.
        symptoms_input: list of strings (cleaned symptoms)
        """
        if not self.model or self.df is None:
            return [{"disease": "System Initializing", "probability": 0, "severity": "Low", "matched_symptoms": []}]

        # Prepare input vector
        # Map input symptoms to feature columns
        feature_cols = self.df.drop("Disease", axis=1).columns
        X_input = np.zeros(len(feature_cols))
        
        matched_symptoms_clean = []
        
        for i, col in enumerate(feature_cols):
            # We assume symptoms_input are already cleaned/fuzzy-matched to match columns
            if col in symptoms_input:
                X_input[i] = 1
                matched_symptoms_clean.append(col)

        X_input = X_input.reshape(1, -1)
        
        try:
            probs = self.model.predict_proba(X_input)[0]
        except Exception as e:
            print(f"Prediction Error: {e}")
            return [{"disease": "Error", "probability": 0, "severity": "Low"}]

        top_indices = np.argsort(probs)[-top_n:][::-1]
        results = []

        for idx in top_indices:
            disease = self.encoder.inverse_transform([idx])[0]
            probability = probs[idx] # 0-1
            
            # Filter low probability noise
            if probability < 0.05:
                continue

            # Severity logic (User specific: match_ratio >= 0.8 is High)
            try:
                disease_row = self.df[self.df['Disease'] == disease].iloc[0]
                total_symptoms = sum(disease_row[1:]) 
                
                # Check which of the USER's symptoms actually match this specific disease
                # (Intersection of user inputs and disease-positive columns)
                matched_for_this_disease = [s for s in matched_symptoms_clean if disease_row[s] == 1]
                symptom_count = len(matched_for_this_disease)
                
                match_ratio = symptom_count / total_symptoms if total_symptoms else 0

                if match_ratio >= 0.8:
                    severity = "High"
                elif match_ratio >= 0.5:
                    severity = "Moderate"
                else:
                    severity = "Low"
                    
                results.append({
                    "disease": disease,
                    "probability": round(probability * 100, 2), # User wanted percentage
                    "severity": severity,
                    "matched_symptoms": matched_for_this_disease # List of strings
                })
            except Exception as e:
                print(f"Severity Calc Error for {disease}: {e}")
                continue

        return results

    def get_precautions_from_csv(self, disease):
        """
        Fetches precautions from the loaded CSV.
        """
        # Reload to capture updates (since dynamic learner updates it)
        if os.path.exists(PRECAUTIONS_PATH):
            temp_df = pd.read_csv(PRECAUTIONS_PATH)
            # Fuzzy or exact match for disease
            # User's CSV has columns: Disease, Symptom, Precaution
            # We want precautions for the Disease generally
            rows = temp_df[temp_df['Disease'] == disease]
            if not rows.empty:
                return rows['Precaution'].unique().tolist()
        return []

    def generate_advisory(self, disease, weighted_score, precautions_list, aq_data=None):
        """
        Generates the text advisory.
        """
        risk_label = "LOW"
        if weighted_score > 40: risk_label = "MODERATE"
        if weighted_score > 70: risk_label = "HIGH"
        if weighted_score > 85: risk_label = "CRITICAL"

        advisory_parts = []
        advisory_parts.append(f"### Analysis for **{disease}**")
        advisory_parts.append(f"**Overall Risk Level:** {risk_label} ({weighted_score:.1f}%)")
        
        if aq_data:
            advisory_parts.append(f"**Environmental Context:** Air Quality is {aq_data.get('status', 'Unknown')} (Score: {aq_data.get('risk_score', 0):.2f})")

        if precautions_list:
            advisory_parts.append("\n**Recommended Precautions:**")
            for p in precautions_list:
                advisory_parts.append(f"- {p}")
        else:
            advisory_parts.append("\n*Precautions are being fetched... check back shortly.*")
            
        return "\n".join(advisory_parts)
