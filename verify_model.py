import os
import sys
from catboost import CatBoostClassifier

# Path setup
sys.path.append(os.getcwd())
model_path = r'd:\ML AI321- FALL 25\ML_PROJECT_HEALTHCARE_AAAHH\disease_catboost_symptoms\models\catboost_model.cbm'

def verify():
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return

    model = CatBoostClassifier()
    model.load_model(model_path)
    
    print("Model loaded successfully.")
    
    # Check for feature names
    features = []
    if hasattr(model, 'feature_names_'):
        print("Found 'feature_names_' attribute.")
        features = model.feature_names_
    elif hasattr(model, 'get_feature_names'):
        print("Found 'get_feature_names()' method.")
        features = model.get_feature_names()
    else:
        print("Could not find feature names method/attribute.")
        
    print(f"Feature count: {len(features)}")
    print(f"Features: {features}")
    
    # Test Prediction validity
    if features:
        # Create dummy vector of correct length
        vec = [0] * len(features)
        try:
            print(f"Testing prediction with vector length {len(vec)}...")
            res = model.predict([vec])
            print(f"Prediction success. Result: {res[0]}")
        except Exception as e:
            print(f"Prediction failed with correct length: {e}")
            
        # Test Fail case (Short vector)
        short_vec = [0] * (len(features) - 1)
        try:
            print(f"Testing prediction with SHORT vector length {len(short_vec)}...")
            model.predict([short_vec])
            print("Prediction unexpectedly succeeded with short vector.")
        except Exception as e:
            print(f"Prediction correctly failed with short vector: {e}")

if __name__ == "__main__":
    verify()
