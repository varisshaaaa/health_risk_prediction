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
from backend.utils.dynamic_learner import integrate_new_symptom
from backend.database import engine, Base, get_db, check_and_migrate_tables
from backend.models import PredictionLog, SymptomLog, Precaution

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Health Advisory API V4.0 (Smart Learning)")
print("--- STARTING APP v4.0: SMART LEARNING ENABLED ---")

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
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "disease_model.pkl")):
        print("Model not found. Triggering initial training...")
        from backend.utils.dynamic_learner import train_model
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
    symptoms: Union[List[str], str] # Can be a list or a raw string
    other_symptoms: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "Health Advisory System V4.0"}

def handle_new_symptoms_bg(new_symptoms: List[str]):
    """Background task to integrate new symptoms"""
    if not new_symptoms:
        return
    print(f"Background: Integrating new symptoms: {new_symptoms}")
    for sym in new_symptoms:
        integrate_new_symptom(sym)
    # Reload orchestrator resources to reflect changes
    orchestrator.load_resources()

@app.post("/predict")
def predict_health_risk(request: PredictionRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # 1. Symptom Parsing & Cleaning
    raw_input = request.symptoms
    if isinstance(raw_input, list):
        raw_input = ", ".join(raw_input) # specific list to string for our parser
    
    if request.other_symptoms:
        raw_input += f", {request.other_symptoms}"

    # Get existing columns to match against
    if orchestrator.df is not None:
        existing_cols = orchestrator.df.drop("Disease", axis=1).columns.tolist()
    else:
        existing_cols = []

    matched_symptoms, new_symptoms = clean_and_extract_smart(raw_input, existing_cols)
    
    # Trigger Dynamic Learning (Background)
    if new_symptoms:
        # Pass native python list
        background_tasks.add_task(handle_new_symptoms_bg, list(new_symptoms))

    # 2. Disease Prediction (Weighted & Severity Logic)
    try:
        # Get top prediction
        predictions = orchestrator.predict_diseases(matched_symptoms, top_n=1)
        
        if not predictions:
            # Fallback if no predictions
            best_pred = {"disease": "Unknown Condition", "probability": 10, "severity": "Low", "matched_symptoms": []}
        else:
            best_pred = predictions[0]

        disease = best_pred['disease']
        symptom_probability = best_pred['probability'] # 0-100
        matched_list = best_pred.get('matched_symptoms', [])
    except Exception as e:
        print(f"Prediction logic error: {e}")
        disease = "Error in Calculation"
        symptom_probability = 0
        best_pred = {"severity": "Low"}
        matched_list = []

    # 3. Demographic Risk
    demo_risk_norm = calculate_demographic_risk(request.age, request.gender) # 0-1
    demo_risk_score = demo_risk_norm * 100

    # 4. Air Quality Risk
    aq_data = get_air_quality_risk(request.city)
    aq_risk_norm = aq_data.get('risk_score', 0) # 0-1
    aq_risk_score = aq_risk_norm * 100

    # 5. Weighted Formula (User Request: Symptom > Demo > Air)
    # "avg ut the threee scores ... give more weightage to symptomscore then demographics then air quality"
    final_score = (symptom_probability * 0.5) + (demo_risk_score * 0.3) + (aq_risk_score * 0.2)
    
    # "check it before using it... check if not available for any symptom then it should be fetched"
    # We use the orchestrator to fetch from CSV. 
    # If missing, we trigger background fetch (handled in dynamic learner logic generally, 
    # OR we can do a specific check here)
    
    # 6. Precautions
    # Fetch precautions (checks CSV first, then triggers scrape if missing)
    precautions_list = orchestrator.get_precautions_from_csv(disease)
    
    # 7. Generate Advisory
    # Returns (advisory_text, risk_label)
    advisory_text, risk_label_from_advisory = orchestrator.generate_advisory(
        disease, final_score, matched_list, precautions_list, aq_data
    )

    # 8. Log to DB
    # Fix: Convert numpy integers/floats to Python native types to avoid Postgres "schema np does not exist" error
    try:
        log_entry = PredictionLog(
            age=int(request.age), # Ensure int
            gender=str(request.gender),
            city=str(request.city),
            symptoms_vector=[], 
            symptom_names=[str(s) for s in matched_list], # Ensure list of strings
            
            symptom_risk=float(symptom_probability/100), # Ensure float
            demographic_risk=float(demo_risk_norm),      # Ensure float
            air_quality_risk=float(aq_risk_norm),        # Ensure float
            
            predicted_disease=str(disease),
            risk_score=float(final_score),               # Ensure float
            risk_level=str(risk_label_from_advisory)     # Use the label from advisory logic
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"DB Logging Error (Non-fatal): {e}")
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

@app.get("/logs/performance")
def get_performance_logs():
    """Returns model training history for visualization"""
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_history.csv")
    if os.path.exists(history_path):
        try:
            df = pd.read_csv(history_path)
            # Return as list of dicts
            return df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return []

@app.get("/symptoms")
def get_symptoms():
    if orchestrator.df is not None:
        return orchestrator.df.drop("Disease", axis=1).columns.tolist()
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
