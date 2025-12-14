import requests
import os
# Replace with your OpenWeatherMap API key
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable not set!")

def fetch_weather_aqi(city_name: str) -> dict:
    """Fetch weather & AQI data for a city."""
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    geo_data = requests.get(geo_url, timeout=5).json()
    if not geo_data:
        raise ValueError(f"City '{city_name}' not found!")

    lat, lon = geo_data[0]['lat'], geo_data[0]['lon']

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    weather_data = requests.get(weather_url, timeout=5).json()

    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    aqi_data = requests.get(aqi_url, timeout=5).json()

    aqi_class = aqi_data['list'][0]['main']['aqi']
    components = aqi_data['list'][0]['components']

    return {
        'AQI_Class': aqi_class,
        'PM10': components.get('pm10', 50),
        'PM2_5': components.get('pm2_5', 25),
        'NO2': components.get('no2', 20),
        'SO2': components.get('so2', 10),
        'O3': components.get('o3', 30),
        'Temperature': weather_data['main']['temp'],
        'Humidity': weather_data['main']['humidity'],
        'WindSpeed': weather_data['wind']['speed']
    }


def age_gender_risk_percent(age: int, gender: str) -> float:
    """Calculate age/gender risk score."""
    gender = gender.lower()
    if age <= 12:
        base_risk = 55
    elif age <= 25:
        base_risk = 25
    elif age <= 39:
        base_risk = 30
    elif age <= 49:
        base_risk = 45
    elif age <= 59:
        base_risk = 60
    elif age <= 69:
        base_risk = 75
    else:
        base_risk = 85

    gender_adjust = 5 if gender == "male" else (-5 if age < 60 else 0)
    risk = base_risk + gender_adjust
    return max(10, min(95, risk))
