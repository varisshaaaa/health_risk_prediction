from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Optional
import sys
import os
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.fetch_air_quality import get_air_quality_risk
from backend.utils.demographic_risk import calculate_demographic_risk
from backend.utils.disease_prediction import DiseaseRiskOrchestrator
from backend.utils.retrain import run_retraining_job
from backend.database import engine, Base, get_db, check_and_migrate_tables
from backend.models import PredictionLog, SymptomLog, Precaution

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Health Advisory API")
print("--- STARTING APP v3.1: WITH AUTO-MIGRATION ---")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Orchestrator (Loads model once)
try:
    orchestrator = DiseaseRiskOrchestrator()
except Exception as e:
    print(f"Warning: Model not found or error loading. Train model first. {e}")
    orchestrator = None

# Scheduler for Retraining
scheduler = BackgroundScheduler()

@app.on_event("startup")
def startup_tasks():
    # 1. Migrate DB
    check_and_migrate_tables()
    
    # 2. Start Scheduler
    scheduler.add_job(run_retraining_job, 'interval', hours=5, id='retrain_model')
    scheduler.start()
    print("Scheduler started: Model retraining set for every 5 hours.")

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

class PredictionRequest(BaseModel):
    age: int
    gender: str
    city: str
    symptoms: List[int] # Binary vector
    symptom_names: List[str] # For explanation
    other_symptoms: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "Health Advisory System API v3.0"}

@app.post("/predict")
def predict_health_risk(request: PredictionRequest, db: Session = Depends(get_db)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model.")
    
    # 1. Air Quality Risk (20%)
    aq_data = get_air_quality_risk(request.city)
    aq_risk_norm = aq_data['risk_score'] # 0.0 - 1.0

    # 2. Symptom Prediction (50%)
    if sum(request.symptoms) == 0:
        disease = "No Specific Disease Detected"
        symptom_risk_norm = 0.1
        pred_result = {"disease": disease, "confidence": 0.1, "risk_level": "LOW"}
    else:
        pred_result = orchestrator.predictor.predict(request.symptoms)
        disease = pred_result['disease']
        symptom_risk_norm = pred_result['confidence'] # 0.0 - 1.0
    
    # 3. Demographic Risk (30%)
    demo_risk_norm = calculate_demographic_risk(request.age, request.gender) # 0.0 - 1.0
    
    # 4. Weighted Aggregation (0-100 Scale)
    # Symptom: 50%, Age/Gender: 30%, Environmental: 20%
    weighted_score = (
        (symptom_risk_norm * 0.5) + 
        (demo_risk_norm * 0.3) + 
        (aq_risk_norm * 0.2)
    )
    final_risk_score = weighted_score * 100 # Convert to 0-100
    
    # Risk Classification
    if final_risk_score < 30:
        risk_label = "LOW"
    elif final_risk_score < 60:
        risk_label = "MODERATE"
    elif final_risk_score < 85:
        risk_label = "HIGH"
    else:
        risk_label = "CRITICAL"

    # 5. Fetch Precautions from DB
    precautions_query = db.query(Precaution).filter(Precaution.disease.ilike(disease.replace("_", " ")))
    
    # Filter based on Severity/Risk
    # Low -> Basic
    # Moderate -> Basic, Moderate
    # High -> Basic, Moderate, Important
    # Critical -> All
    
    if risk_label == "LOW":
        target_severities = ["BASIC"]
    elif risk_label == "MODERATE":
        target_severities = ["BASIC", "MODERATE"]
    elif risk_label == "HIGH":
        target_severities = ["BASIC", "MODERATE", "IMPORTANT"]
    else:
        target_severities = ["BASIC", "MODERATE", "IMPORTANT", "URGENT"]
        
    # Note: If no severity labels in DB (imported data might vary), fallback to showing all or limit count
    # Ideally, scraping classifies them.
    
    precautions_list = []
    # If we have severity logic in DB
    precautions_objs = precautions_query.filter(Precaution.severity_level.in_(target_severities)).all()
    if not precautions_objs:
        # Fallback: Just get all for disease if filtering resulted in empty (or if severities missing)
        precautions_objs = precautions_query.limit(5).all()
        
    precautions_text = "\n".join([f"- {p.content}" for p in precautions_objs])
    
    # Fallback if DB empty (use CSV logic from orchestrator as backup-backup)
    if not precautions_text:
        legacy_text = orchestrator.get_precautions(disease, weighted_score)
        precautions_text = legacy_text

    # 6. Generate Advisory (Validation Logic)
    advisory, _ = orchestrator.generate_advisory(
        disease, weighted_score, request.symptom_names, precautions_text, aq_data=aq_data
    )
    
    # --- DB LOGGING ---
    feature_log = PredictionLog(
        age=request.age,
        gender=request.gender,
        city=request.city,
        symptoms_vector=request.symptoms,
        symptom_names=request.symptom_names,
        
        # Detailed Components
        symptom_risk=symptom_risk_norm,
        demographic_risk=demo_risk_norm,
        air_quality_risk=aq_risk_norm,
        
        predicted_disease=disease,
        risk_score=final_risk_score,
        risk_level=risk_label
    )
    db.add(feature_log)
    
    if request.other_symptoms:
        new_symptom = SymptomLog(symptom_text=request.other_symptoms)
        db.add(new_symptom)
        
    db.commit()
    
    return {
        "disease": disease,
        "risk_score": final_risk_score,
        "risk_level": risk_label,
        "components": {
            "symptom_contribution": symptom_risk_norm * 0.5 * 100,
            "demographic_contribution": demo_risk_norm * 0.3 * 100,
            "environmental_contribution": aq_risk_norm * 0.2 * 100
        },
        "air_quality": aq_data,
        "advisory": advisory,
        "precautions": [p.content for p in precautions_objs] if precautions_objs else ["Consult a doctor."]
    }

@app.get("/logs/features")
def get_feature_logs(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(50).all()
    return logs

@app.get("/logs/symptoms")
def get_symptom_logs(db: Session = Depends(get_db)):
    logs = db.query(SymptomLog).order_by(SymptomLog.timestamp.desc()).all()
    return logs

@app.get("/symptoms")
def get_symptoms():
    if orchestrator and orchestrator.predictor:
        features = orchestrator.predictor.get_feature_names()
        if features:
            return features
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'disease_catboost_symptoms', 'data', 'clean_symptoms.csv')
        df = pd.read_csv(csv_path)
        symptoms = [col for col in df.columns if col.lower() != 'disease']
        return symptoms
    except Exception as e:
        print(f"Error loading symptoms: {e}")
        return ["fever", "cough", "fatigue"]

@app.post("/admin/retrain")
def trigger_retrain():
    """Manual trigger for retraining"""
    run_retraining_job()
    return {"status": "Retraining triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
