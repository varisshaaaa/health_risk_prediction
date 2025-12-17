from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import sys

# Services
from backend.services.health_features import get_air_quality_risk
from backend.services.demographic_risk import calculate_demographic_risk
from backend.services.symptom_processor import SymptomProcessor
from backend.services.disease_prediction import DiseaseRiskOrchestrator
from backend.services.dynamic_learning import integrate_new_symptom
from backend.database.database import SessionLocal
from backend.database.models import PredictionLog

router = APIRouter()

# Initialize Services
disease_orchestrator = DiseaseRiskOrchestrator()
known_symptoms = []
if disease_orchestrator.df is not None:
    known_symptoms = [c for c in disease_orchestrator.df.columns if c != 'Disease']
symptom_processor = SymptomProcessor(known_symptoms)

class PredictRequest(BaseModel):
    age: int
    gender: str
    city: str
    checked_symptoms: list[str] = []
    free_text_symptoms: str = ""

async def integrate_and_reload(symptom):
    """
    Integrates new symptom and reloads the model in orchestrator.
    """
    success = integrate_new_symptom(symptom)
    if success:
        # Reload orchestrator resources
        disease_orchestrator.load_resources()
        # Update processor known symptoms
        if disease_orchestrator.df is not None:
             symptom_processor.known_symptoms = [c for c in disease_orchestrator.df.columns if c != 'Disease']

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "Integrated Health Backend"}

@router.post("/predict")
async def predict_health_risk(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Unified Endpoint:
    1. Features (AQI, etc.)
    2. Demographic Risk
    3. Disease Prediction (Symptoms)
    4. Combined Weighted Risk
    """
    
    # 1. Health Impact / Environmental Features
    aq_data = get_air_quality_risk(request.city)
    health_impact_score = aq_data.get("risk_score", 0.2) # 0-1
    
    # 2. Demographic Risk
    demo_risk = calculate_demographic_risk(request.age, request.gender) # 0-1
    
    # 3. Symptom Processing & Disease Prediction
    # Combine checked + free text
    matched, new_candidates = symptom_processor.extract_symptoms(request.checked_symptoms, request.free_text_symptoms)
    
    # Predict Disease
    disease_results = disease_orchestrator.predict_diseases(matched, top_n=3)
    top_disease = disease_results[0]
    
    # Symptom Risk Score (Based on top disease probability & severity)
    symptom_risk_score = 0.0
    if top_disease["disease"] != "System Initializing":
        prob = top_disease["probability"] / 100.0
        severity_map = {"High": 1.0, "Moderate": 0.6, "Low": 0.3}
        sev_score = severity_map.get(top_disease["severity"], 0.3)
        symptom_risk_score = (prob * 0.7) + (sev_score * 0.3) # Weighted
    
    # 4. Overall Weighted Risk Calculation
    overall_risk = (0.5 * symptom_risk_score) + (0.3 * demo_risk) + (0.2 * health_impact_score)
    overall_risk_percent = round(overall_risk * 100, 2)
    
    risk_level = "Low"
    if overall_risk_percent > 40: risk_level = "Moderate"
    if overall_risk_percent > 70: risk_level = "High"
    
    # Precautions
    precautions = []
    if top_disease["disease"] != "System Initializing":
        precautions = disease_orchestrator.get_precautions(top_disease["disease"])
    
    # Dynamic Learning: Handle new symptoms in background
    if new_candidates:
        print(f"New symptoms detected: {new_candidates}")
        for s in new_candidates:
            background_tasks.add_task(integrate_and_reload, s)

    # Log Prediction
    try:
        db = SessionLocal()
        if db:
            log_entry = PredictionLog(
                age=request.age,
                gender=request.gender,
                city=request.city,
                symptoms_vector=matched,
                symptom_names=matched,
                symptom_risk=symptom_risk_score,
                demographic_risk=demo_risk,
                air_quality_risk=health_impact_score,
                predicted_disease=top_disease["disease"],
                risk_score=overall_risk,
                risk_level=risk_level
            )
            db.add(log_entry)
            db.commit()
            db.close()
    except Exception as e:
        print(f"Logging failed: {e}")

    return {
        "predicted_disease": top_disease["disease"],
        "probability": top_disease["probability"],
        "severity": top_disease["severity"],
        "overall_health_risk": overall_risk_percent,
        "risk_level": risk_level,
        "matched_symptoms": matched,
        "precautions": precautions,
        "environmental_data": aq_data,
        "breakdown": {
            "symptom_contribution": round(0.5 * symptom_risk_score * 100, 1),
            "demographic_contribution": round(0.3 * demo_risk * 100, 1),
            "environmental_contribution": round(0.2 * health_impact_score * 100, 1)
        }
    }
