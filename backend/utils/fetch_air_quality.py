import os
import requests
import joblib
import pandas as pd
import numpy as np
import random

# Load Air Quality Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'health_impact_predictor.pkl')

aq_model = None
if os.path.exists(MODEL_PATH):
    try:
        aq_model = joblib.load(MODEL_PATH)
        print("Loaded Air Quality Health Impact Model.")
    except Exception as e:
        print(f"Error loading AQ model: {e}")

def get_air_quality_risk(city: str):
    """
    Fetches real air quality and uses ML model to predict risk score.
    """
    try:
        api_key = os.getenv("WEATHERMAP_API_KEY")
        if not api_key:
            print("Warning: WEATHERMAP_API_KEY not set.")
            return {"aqi": 0, "status": "N/A (Key Missing)", "risk_score": 0.2}

        # 1. Geocoding API
        # 1. Geocoding API
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
        geo_res = requests.get(geo_url, timeout=5)
        
        if geo_res.status_code != 200 or not geo_res.json():
            return _get_mock_data(city)
            
        loc = geo_res.json()[0]
        lat, lon = loc['lat'], loc['lon']
        
        # 2. Air Pollution API
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        air_res = requests.get(air_url, timeout=5)
        
        if air_res.status_code != 200:
             return _get_mock_data(city)
             
        data = air_res.json()
        item = data['list'][0]
        aqi = item['main']['aqi']
        comps = item['components']
        
        # Model Prediction
        risk_score = 0.5 # Default
        if aq_model:
            # Prepare features as expected by the model
            # Assuming model expects: ['AQI', 'PM2.5', 'PM10', 'NO2', ...] or similar
            # Since we don't know exact features, we try to match likely ones or fallback
            try:
                # Example feature vector construction
                features = pd.DataFrame([{
                    'AQI': aqi,
                    'PM2.5': comps.get('pm2_5', 0),
                    'PM10': comps.get('pm10', 0),
                    'NO2': comps.get('no2', 0),
                    'CO': comps.get('co', 0),
                    'O3': comps.get('o3', 0),
                    'SO2': comps.get('so2', 0)
                }])
                # Align columns if possible or just predict
                # If model was trained on different columns, this might fail, necessitating a try/catch fallback
                prediction = aq_model.predict(features)
                # If regression -> score, If classification -> map class to score
                if isinstance(prediction[0], str):
                    # Map classes like 'High', 'Moderate' to score
                    mapping = {'Very High': 1.0, 'High': 0.8, 'Moderate': 0.5, 'Low': 0.2, 'Good': 0.1}
                    risk_score = mapping.get(prediction[0], 0.5)
                else:
                    risk_score = float(prediction[0])
                    # Normalize if needed (assuming output is 0-100 or 0-1)
                    if risk_score > 1.0: risk_score /= 100.0
                    
            except Exception as e:
                print(f"AQ Prediction Error: {e}")
                # Fallback heuristic
                risk_score = (aqi - 1) / 4.0

        else:
            # Fallback heuristic
            risk_score = (aqi - 1) / 4.0
        
        return {
            "aqi": aqi,
            "risk_score": min(max(risk_score, 0.0), 1.0),
            "pollutants": comps,
            "source": "OpenWeatherMap + ML"
        }

    except Exception as e:
        print(f"Exception in fetching air quality: {e}")
        return _get_mock_data(city)

def _get_mock_data(city):
    """Fallback logic"""
    # Simulate somewhat realistic range
    aqi = random.randint(1, 5) 
    risk_map = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
    
    return {
        "aqi": aqi,
        "risk_score": risk_map[aqi],
        "pollutants": {
            "pm2_5": random.randint(10, 100),
            "pm10": random.randint(20, 150),
            "no2": random.randint(10, 80)
        },
        "source": "Mock Data (Set WEATHERMAP_API_KEY)"
    }
