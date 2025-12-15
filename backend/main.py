from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Optional
import sys
import os
from sqlalchemy.orm import Session

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.fetch_air_quality import get_air_quality_risk
from backend.utils.demographic_risk import calculate_demographic_risk
from backend.utils.disease_prediction import DiseaseRiskOrchestrator
from backend.database import engine, Base, get_db
from backend.models import PredictionLog, SymptomLog

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Health Advisory API")

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

class PredictionRequest(BaseModel):
    age: int
    gender: str
    city: str
    symptoms: List[int] # Binary vector
    symptom_names: List[str] # For explanation
    other_symptoms: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "Health Advisory System API"}

@app.post("/predict")
def predict_health_risk(request: PredictionRequest, db: Session = Depends(get_db)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model.")
    
    # 3. Air Quality Risk (Moved up to capture feature for DB)
    aq_data = get_air_quality_risk(request.city)
    aq_risk = aq_data['risk_score']

    # 1. Symptom Prediction
    pred_result = orchestrator.predictor.predict(request.symptoms)
    disease = pred_result['disease']
    symptom_risk = pred_result['confidence']
    
    # 2. Demographic Risk
    demo_risk = calculate_demographic_risk(request.age, request.gender)
    
    # 4. Weighted Aggregation
    final_risk = (0.6 * symptom_risk) + (0.3 * demo_risk) + (0.1 * aq_risk)
    
    # 5. Generate Advice
    precautions = orchestrator.get_precautions(disease, final_risk)
    advisory, risk_label = orchestrator.generate_advisory(disease, final_risk, request.symptom_names, precautions)
    
    # --- DB LOGGING (Feature Store) ---
    feature_log = PredictionLog(
        age=request.age,
        gender=request.gender,
        city=request.city,
        symptoms_vector=request.symptoms,
        symptom_names=request.symptom_names,
        air_quality_risk=aq_risk,
        predicted_disease=disease,
        risk_score=final_risk,
        risk_level=risk_label
    )
    db.add(feature_log)
    
    # Log new symptoms
    if request.other_symptoms:
        # Check if exists or just append
        new_symptom = SymptomLog(symptom_text=request.other_symptoms)
        db.add(new_symptom)
        
    db.commit()
    
    return {
        "disease": disease,
        "risk_score": final_risk,
        "risk_level": risk_label,
        "air_quality": aq_data,
        "advisory": advisory,
        "symptom_match_details": pred_result
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
    """
    Returns the list of symptoms required by the model.
    Fetches directly from the loaded model to ensure compatibility.
    """
    if orchestrator and orchestrator.predictor:
        features = orchestrator.predictor.get_feature_names()
        if features:
            return features
            
    # Fallback to CSV if model features not available or model not loaded
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'disease_catboost_symptoms', 'data', 'clean_symptoms.csv')
        df = pd.read_csv(csv_path)
        # Columns are symptoms + 'disease'
        symptoms = [col for col in df.columns if col.lower() != 'disease']
        return symptoms
    except Exception as e:
        print(f"Error loading symptoms: {e}")
        return ["fever", "cough", "fatigue", "headache", "nausea", "skin_rash", "joint_pain"]

@app.get("/monitoring/drift")
def check_data_drift(db: Session = Depends(get_db)):
    """
    Simple Data Drift Check: Compares recent production data (last 100 entries) 
    vs Training Data stats (Baseline).
    """
    report = {"status": "stable", "drift_detected": False, "details": {}}
    
    # 1. Fetch Production Data
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(100).all()
    if not logs:
        return {"status": "insufficient_data", "message": "Not enough production data to calculate drift."}
    
    # Convert to DataFrame
    prod_data = [{"age": log.age, "gender": log.gender} for log in logs]
    df_prod = pd.DataFrame(prod_data)
    
    # 2. Compare Mean Age (Simple Statistical Check)
    # Baseline (hardcoded from training analysis or loaded)
    baseline_mean_age = 30.0 # Example baseline
    current_mean_age = df_prod['age'].mean()
    
    diff = abs(current_mean_age - baseline_mean_age)
    report['details']['age_drift'] = {
        "baseline": baseline_mean_age,
        "current": current_mean_age,
        "difference": diff
    }
    
    if diff > 10: # Threshold
        report['drift_detected'] = True
        report['status'] = "warning"
        report['details']['alert'] = "Significant drift in 'Age' distribution detected."
        
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
