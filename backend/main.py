from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from app.model import model, imputer, scaler, features
from app.services import fetch_weather_aqi, age_gender_risk_percent

app = FastAPI(title="Health Risk Prediction API")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class PredictRequest(BaseModel):
    city: str
    age: int
    gender: str

class PredictResponse(BaseModel):
    health_score: float
    age_gender_score: float
    final_score: float
    risk_label: str

# ---------- Risk Classification ----------
def classify_risk(score: float) -> str:
    if score >= 85:
        return "🔴 VERY HIGH RISK"
    elif score >= 70:
        return "🟠 HIGH RISK"
    elif score >= 50:
        return "🟡 MEDIUM RISK"
    elif score >= 30:
        return "🟢 LOW RISK"
    else:
        return "🟢 VERY LOW RISK"

# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "Health Risk Prediction API. Use /predict endpoint."}

# ---------- Prediction ----------
@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        weather_input = fetch_weather_aqi(data.city)
        input_df = pd.DataFrame([weather_input])[features]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        health_score = model.predict(input_scaled)[0]
        age_gender_score = age_gender_risk_percent(data.age, data.gender)
        final_score = 0.6 * age_gender_score + 0.4 * health_score
        risk_label = classify_risk(final_score)

        return PredictResponse(
            health_score=round(health_score, 2),
            age_gender_score=round(age_gender_score, 2),
            final_score=round(final_score, 2),
            risk_label=risk_label
        )
    except Exception as e:
        return {"error": str(e)}
