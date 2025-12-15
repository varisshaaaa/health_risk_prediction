import os
import requests
import random

def get_air_quality_risk(city: str):
    """
    Fetches real air quality from OpenWeatherMap.
    Falls back to mock data if API key is missing or call fails.
    """
    api_key = os.getenv("WEATHERMAP_API_KEY")
    
    if not api_key:
        print("Warning: WEATHERMAP_API_KEY not set. Using mock data.")
        return _get_mock_data(city)

    try:
        # 1. Geocoding API to get Lat/Lon
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
        geo_res = requests.get(geo_url, timeout=5)
        
        if geo_res.status_code != 200 or not geo_res.json():
            print(f"Error fetching location for {city}. Using mock data.")
            return _get_mock_data(city)
            
        location_data = geo_res.json()[0]
        lat, lon = location_data['lat'], location_data['lon']
        
        # 2. Air Pollution API
        # http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API key}
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        air_res = requests.get(air_url, timeout=5)
        
        if air_res.status_code != 200:
             print(f"Error fetching air quality. Status: {air_res.status_code}")
             return _get_mock_data(city)
             
        data = air_res.json()
        
        # AQI is 1 (Good) to 5 (Very Poor)
        aqi = data['list'][0]['main']['aqi']
        components = data['list'][0]['components']
        
        # Normalize risk: 
        # AQI 1 = 0.0, AQI 5 = 1.0 (Linear approx: (aqi-1)/4 )
        # Or simple mapping:
        risk_map = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
        risk_score = risk_map.get(aqi, 0.5)
        
        return {
            "aqi": aqi,
            "risk_score": risk_score,
            "pollutants": {
                "pm2_5": components.get('pm2_5', 0),
                "pm10": components.get('pm10', 0),
                "no2": components.get('no2', 0)
            },
            "source": "OpenWeatherMap"
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
