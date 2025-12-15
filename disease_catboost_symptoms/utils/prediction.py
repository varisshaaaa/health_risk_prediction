import os
from catboost import CatBoostClassifier
import pandas as pd

class SymptomPredictor:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models', 'catboost_model.cbm')
        self.model = CatBoostClassifier()
        self.load_model()

    def load_model(self):
        if os.path.exists(self.MODEL_PATH):
            self.model.load_model(self.MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model not found at {self.MODEL_PATH}. Please train the model first.")

    def predict(self, symptoms_vector):
        """
        symptoms_vector: list or array of binary values representing symptoms
        """
        prediction = self.model.predict([symptoms_vector])[0][0]
        probabilities = self.model.predict_proba([symptoms_vector])[0]
        confidence = max(probabilities)
        
        # Simple risk level logic based on confidence or specific diseases (can be enhanced)
        risk_level = "MEDIUM" if confidence < 0.7 else "HIGH"
        
        return {
            "disease": prediction,
            "confidence": float(confidence),
            "risk_level": risk_level
        }

if __name__ == "__main__":
    # Test
    # predictor = SymptomPredictor()
    # print(predictor.predict([1,1,1,1,0,0,0]))
    pass
