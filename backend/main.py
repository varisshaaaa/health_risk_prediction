from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
