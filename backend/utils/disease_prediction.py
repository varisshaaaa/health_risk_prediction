import sys
import os
import pandas as pd

# Add project root to sys.path to allow sibling imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from disease_catboost_symptoms.utils.prediction import SymptomPredictor


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

    def generate_advisory(self, disease, final_risk_score, symptoms_list, precautions_text, aq_data=None):
        risk_label = "LOW"
        if final_risk_score > 0.4: risk_label = "MEDIUM"
        if final_risk_score > 0.7: risk_label = "HIGH"

        # 1. Base Disease Advice
        advisory_parts = []
        advisory_parts.append(f"Based on your inputs, the system indicates a potential risk of **{disease}**.")
        advisory_parts.append(f"Your calculated risk level is **{risk_label}** ({final_risk_score:.2f}).")
        
        # 2. Smart Contextual Advice (NLP-like)
        smart_tips = []
        
        # Air Quality Context
        if aq_data:
            aqi = aq_data.get('aqi', 0)
            if aqi > 3: # Poor/Very Poor
                smart_tips.append("⚠️ **Air Quality Alert**: The air quality is currently poor. Wear a mask outdoors and use an air purifier if possible.")
            elif aqi == 1:
                smart_tips.append("✅ Air quality is good, which supports recovery.")

        # Symptom/Disease Specific Context
        disease_lower = disease.lower().replace("_", " ")
        if "cold" in disease_lower or "flu" in disease_lower:
            smart_tips.append("💧 **Stay Hydrated**: Drink plenty of warm fluids.")
            smart_tips.append("🛌 **Rest**: Ensure you get adequate sleep to boost your immune system.")
        elif "migraine" in disease_lower:
            smart_tips.append("🕶️ **Avoid Triggers**: Stay in a quiet, dark room and avoid bright lights.")
        elif "typhoid" in disease_lower or "diarrhea" in disease_lower:
            smart_tips.append("🥗 **Hygiene**: Eat only home-cooked, fresh food and drink boiled water.")
            
        # General advice if risk is low but some symptoms present
        if final_risk_score < 0.2 and not symptoms_list:
             advisory_parts = ["You appear to be at **Low Risk**. However, maintaining a healthy lifestyle is always recommended."]

        if smart_tips:
            advisory_parts.append("\n**💡 Smart Health Tips:**")
            advisory_parts.extend([f"- {tip}" for tip in smart_tips])

        # 3. Standard Precautions
        if precautions_text and precautions_text.lower() != "nan":
            cleaned_precautions = precautions_text.replace("_", " ").title()
            advisory_parts.append(f"\n**🛡️ Standard Precautions for {disease.replace('_', ' ')}:**")
            advisory_parts.append(cleaned_precautions)
        
        advisory_parts.append("\n*Please monitor your condition and consult a doctor if symptoms persist.*")
        
        return "\n".join(advisory_parts), risk_label

