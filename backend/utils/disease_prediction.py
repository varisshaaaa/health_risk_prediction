import sys
import os
import pandas as pd

# Add project root to sys.path to allow sibling imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from disease_catboost_symptoms.utils.prediction import SymptomPredictor
from disease_catboost_symptoms.data import clean_precautions

class DiseaseRiskOrchestrator:
    def __init__(self):
        self.predictor = SymptomPredictor()
        # Load precautions
        precautions_path = os.path.join(project_root, 'disease_catboost_symptoms', 'data', 'clean_precautions.csv')
        self.precautions_df = pd.read_csv(precautions_path)

    def get_precautions(self, disease, risk_level):
        row = self.precautions_df[self.precautions_df['Disease'] == disease]
        if row.empty:
            return "Consult a doctor for specific advice."
        
        # specific precautions
        p_list = row.iloc[0, 1:].dropna().tolist()
        return ", ".join(p_list)

    def process_prediction(self, symptoms, age, gender, aq_risk_score):
        # 1. Symptom Prediction
        pred_result = self.predictor.predict(symptoms)
        disease = pred_result['disease']
        conf = pred_result['confidence']
        symptom_risk_score = conf # Use confidence as proxy for risk intensity from symptoms

        # 2. Demographic Risk
        # (Already calculated in main flow, but could be refined here)
        # We'll assume the inputs to this function are raw or pre-calculated risks
        
        # 3. Final Risk Calculation
        # Weights: Symptom (0.6), Demo (0.3), AQ (0.1)
        # Note: 'age' and 'gender' logic handled in demographic_risk.py, expected passed as score? 
        # Let's assume we get the scores.
        
        pass # Logic moved to main or structured here. 
        # Actually, let's keep this class focused on the Disease part and aggregation.

    def generate_advisory(self, disease, final_risk_score, symptoms_list, precautions_text):
        risk_label = "LOW"
        if final_risk_score > 0.4: risk_label = "MEDIUM"
        if final_risk_score > 0.7: risk_label = "HIGH"

        advisory = f"""
        Based on your symptoms ({', '.join(symptoms_list)}), you may have **{disease}**. 
        Your overall risk level is **{risk_label}** ({final_risk_score:.2f}).
        
        **Recommended Precautions:**
        {precautions_text}.
        
        Please monitor your condition.
        """
        return advisory, risk_label

