from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
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
        
        # THRESHOLD LOGIC
        if pred_result['confidence'] < 0.35:
            disease = "No Specific Disease Detected"
            symptom_risk_norm = 0.1 # Cap low
        else:
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

from backend.utils.scrape_and_import import scrape_and_import_disease

    # 5. Fetch Precautions from DB
    precautions_objs = []
    precautions_text = ""
    
    if disease != "No Specific Disease Detected":
        precautions_query = db.query(Precaution).filter(Precaution.disease.ilike(disease.replace("_", " ")))
        
        # Filter based on Severity/Risk
        # Low -> Basic
        # Moderate -> Basic, Moderate
        # High -> Basic, Moderate, Important
        # Critical -> All
        
        target_severities = ["BASIC"]
        if risk_label in ["MODERATE", "HIGH", "CRITICAL"]:
            target_severities.append("MODERATE")
        if risk_label in ["HIGH", "CRITICAL"]:
            target_severities.append("IMPORTANT")
        if risk_label == "CRITICAL":
            target_severities.append("URGENT")
            
        # Try prioritized fetch
        precautions_objs = precautions_query.filter(Precaution.severity_level.in_(target_severities)).all()
        
        # FALLBACK 1: If specific severities missing, get ANY for this disease
        if not precautions_objs:
            precautions_objs = precautions_query.limit(5).all()
            
        # TRIGGER SCRAPING: If still empty, trigger scraping for next time
        if not precautions_objs:
             print(f"SCRAPE TRIGGER: No precautions found for {disease}. Triggering background scrape.")
             # Trigger the scraper in the background
             # Note: 'background_tasks' must be added to endpoint arguments
             # For now, since we didn't add it in func sig, we will just log or assume user adds it
             # BUT WAIT, I am editing the code, I can add it!
             # I need to change the function signature in the next step or handle it here if possible.
             # Actually, since I can't change signature easily in a block replace without rewriting the whole func def,
             # I will use a direct threading call or just rely on the 'missing_diseases.txt' if I can't add BackgroundTasks easily.
             
             # BETTER APPROACH: I will just call the function in a thread to mimic BackgroundTasks if signature change is hard,
             # OR I will just rewrite the function signature in a separate call?
             # Let's try to just use the log file for now as the 'robust' way requested by user might need code changes I can't fully certify without signature change.
             # USER REQUESTED "Complete Web Scraping Flow". Passive log isn't enough.
             
             # I will assume I can update the signature in a subsequent call or simple threading here.
             import threading
             t = threading.Thread(target=scrape_and_import_disease, args=(disease,))
             t.start()
             
             precautions_text = "Fetching specialized precautions from the web... Check back in 30 seconds."
    
    if precautions_objs:
        precautions_text = "\n".join([f"- {p.content}" for p in precautions_objs])
    else:
        # Fallback if DB empty 
        legacy_text = orchestrator.get_precautions(disease, weighted_score)
        if not precautions_text: # Don't overwrite fetching message
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
