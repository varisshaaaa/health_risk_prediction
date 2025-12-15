import random

def get_air_quality_risk(city: str):
    """
    Simulates fetching air quality for a city and returning a risk score (0-1).
    In production, use an API like OpenWeatherMap.
    """
    # Mocking data
    print(f"Fetching air quality for {city}...")
    
    # Simulate different AQI based on city name hash or random
    aqi = random.randint(50, 300) 
    
    # Normalize risk: AQI > 300 is 1.0, AQI < 50 is 0.0
    risk_score = min(aqi / 300, 1.0)
    
    return {
        "aqi": aqi,
        "risk_score": round(risk_score, 2),
        "pollutants": {
            "pm2_5": random.randint(10, 100),
            "pm10": random.randint(20, 150),
            "no2": random.randint(10, 80)
        }
    }
