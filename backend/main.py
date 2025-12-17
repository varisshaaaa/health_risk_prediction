from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Optional, Union
import sys
import os
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NLTK Setup at Import Time
import nltk
nltk_packages = ['punkt', 'stopwords']
for pkg in nltk_packages:
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"Failed to download {pkg}: {e}")

from backend.utils.fetch_air_quality import get_air_quality_risk
from backend.utils.demographic_risk import calculate_demographic_risk
from backend.utils.disease_prediction import DiseaseRiskOrchestrator
from backend.utils.symptom_manager import clean_and_extract_smart
from backend.utils.dynamic_learner import integrate_new_symptom, scrape_precautions_for_disease, update_precautions_db_entry
from backend.database import engine, Base, get_db, check_and_migrate_tables
from backend.models import PredictionLog, SymptomLog

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Health Advisory API V5.0 (Scraper Integrated)")
print("--- STARTING APP v5.0: SCRAPER CONNECTED ---")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Orchestrator
orchestrator = DiseaseRiskOrchestrator()

# Scheduler
scheduler = BackgroundScheduler()

@app.on_event("startup")
def startup_tasks():
    check_and_migrate_tables()
    
    # Check if model exists, if not, train it
    from backend.utils.dynamic_learner import MODEL_PATH, train_model
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Triggering initial training...")
        try:
            train_model()
            orchestrator.load_resources() # Reload after training
        except Exception as e:
            print(f"Initial training failed: {e}")

    print("Application Startup Complete.")

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

class PredictionRequest(BaseModel):
    age: int
    gender: str
    city: str
    symptoms: Union[List[str], str]
    other_symptoms: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "Health Advisory System V5.0"}

def handle_new_symptoms_bg(new_symptoms: List[str]):
    """Background task to integrate new symptoms"""
    if not new_symptoms:
        return
    print(f"Background: Processing potential new symptoms: {new_symptoms}")
    # Logic: Verify -> Update -> Retrain
    updated = False
    for sym in new_symptoms:
        # integrate_new_symptom handles verification and db update and retraining
        success = integrate_new_symptom(sym)
        if success:
            updated = True
    
    if updated:
        # Reload orchestrator resources to reflect new model/columns
        print("Reloading orchestrator after learning...")
        orchestrator.load_resources()

@app.post("/predict")
def predict_health_risk(request: PredictionRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # 1. Symptom Parsing & Cleaning
    raw_input = request.symptoms
    if isinstance(raw_input, list):
        raw_input = ", ".join(raw_input)
    
    if request.other_symptoms:
        raw_input += f", {request.other_symptoms}"

    # Get known symptoms from orchestrator
    if orchestrator.df is not None:
        existing_cols = orchestrator.df.drop("Disease", axis=1).columns.tolist()
    else:
        existing_cols = []

    matched_symptoms, new_symptoms = clean_and_extract_smart(raw_input, existing_cols)
    
    # 2. Trigger Learning for New Symptoms (Background)
    if new_symptoms:
        background_tasks.add_task(handle_new_symptoms_bg, list(new_symptoms))

    # 3. Disease Prediction (Using Model)
    if not matched_symptoms:
        disease = "No Detectable Disease"
        symptom_probability = 0
        best_pred = {"severity": "Low"}
        matched_list = []
        final_score = 0 
    else:
        try:
            predictions = orchestrator.predict_diseases(matched_symptoms, top_n=1)
            
            if not predictions:
                best_pred = {"disease": "Unknown Condition", "probability": 10, "severity": "Low", "matched_symptoms": []}
            else:
                best_pred = predictions[0]

            disease = best_pred['disease']
            symptom_probability = best_pred['probability'] 
            matched_list = best_pred.get('matched_symptoms', [])
        except Exception as e:
            print(f"Prediction logic error: {e}")
            disease = "Error in Calculation"
            symptom_probability = 0
            best_pred = {"severity": "Low"}
            matched_list = []

    # 4. Contextual Risks
    demo_risk_norm = calculate_demographic_risk(request.age, request.gender)
    demo_risk_score = demo_risk_norm * 100
    
    aq_data = get_air_quality_risk(request.city)
    aq_risk_norm = aq_data.get('risk_score', 0)
    aq_risk_score = aq_risk_norm * 100

    # 5. Weighted Score
    final_score = (symptom_probability * 0.5) + (demo_risk_score * 0.3) + (aq_risk_score * 0.2)
    
    # 6. Precautions Logic
    precautions_list = []
    if disease != "No Detectable Disease":
        # Check DB first
        precautions_list = orchestrator.get_precautions_from_csv(disease)
        
        # If missing -> Scrape -> Update DB
        if not precautions_list:
            print(f"Precautions missing for {disease}. Scraping now...")
            
            # Scrape
            new_precautions = scrape_precautions_for_disease(disease)
            
            if new_precautions:
                # Update Precautions DB
                # Schema requires a symptom linkage, we'll likely use the primary matched symptom or "General"
                primary_symptom = matched_list[0] if matched_list else "General"
                update_precautions_db_entry(disease, primary_symptom, new_precautions)
                
                # Assign to current response
                precautions_list = new_precautions
                
                # Ideally reload orchestrator's precaution cache?
                orchestrator.reload_precautions()

    # 7. Generate Advisory
    advisory_text, risk_label_from_advisory = orchestrator.generate_advisory(
        disease, final_score, matched_list, precautions_list, aq_data
    )

    # 8. Log to DB
    try:
        log_entry = PredictionLog(
            age=int(request.age), 
            gender=str(request.gender),
            city=str(request.city),
            symptoms_vector=[], 
            symptom_names=[str(s) for s in matched_list],
            symptom_risk=float(symptom_probability/100),
            demographic_risk=float(demo_risk_norm),      
            air_quality_risk=float(aq_risk_norm),        
            predicted_disease=str(disease),
            risk_score=float(final_score),             
            risk_level=str(risk_label_from_advisory)    
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"DB Logging Error: {e}")
        db.rollback()

    return {
        "disease": disease,
        "overall_health_risk": round(final_score, 2),
        "disease_severity": risk_label_from_advisory,
        "probability": round(symptom_probability, 1),
        "matched_symptoms": matched_list,
        "precautions": precautions_list,
        "air_quality": aq_data,
        "advisory": advisory_text
    }

@app.get("/logs/symptoms")
def get_symptom_logs(db: Session = Depends(get_db)):
    logs = db.query(SymptomLog).order_by(SymptomLog.timestamp.desc()).all()
    return logs

@app.get("/symptoms")
def get_symptoms():
    if orchestrator.df is not None:
        return orchestrator.df.drop("Disease", axis=1).columns.tolist()
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
