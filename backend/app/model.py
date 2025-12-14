import joblib
import os

# Absolute path to your model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "health_impact_predictor.pkl")

# Load model package
model_package = joblib.load(MODEL_PATH)

# Extract items from saved package
model = model_package['model']
imputer = model_package['imputer']
scaler = model_package['scaler']
features = model_package['features']
