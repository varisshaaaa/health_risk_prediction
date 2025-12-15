from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.fetch_air_quality import get_air_quality_risk
from backend.utils.demographic_risk import calculate_demographic_risk
from backend.utils.disease_prediction import DiseaseRiskOrchestrator

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
def predict_health_risk(request: PredictionRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model.")
    
    # Logic to log new symptoms
    if request.other_symptoms:
        # Append to a log file for future analysis/training
        log_path = "new_symptoms_log.csv"
        with open(log_path, "a") as f:
            f.write(f"{request.other_symptoms}\n")

    # 1. Symptom Prediction
    pred_result = orchestrator.predictor.predict(request.symptoms)
    disease = pred_result['disease']
    symptom_risk = pred_result['confidence']
    
    # 2. Demographic Risk
    demo_risk = calculate_demographic_risk(request.age, request.gender)
    
    # 3. Air Quality Risk
    aq_data = get_air_quality_risk(request.city)
    aq_risk = aq_data['risk_score']
    
    # 4. Weighted Aggregation
    # final_risk_score = 0.6 * symptom_risk + 0.3 * demographic_risk + 0.1 * air_quality_risk
    final_risk = (0.6 * symptom_risk) + (0.3 * demo_risk) + (0.1 * aq_risk)
    
    # 5. Generate Advice
    precautions = orchestrator.get_precautions(disease, final_risk)
    advisory, risk_label = orchestrator.generate_advisory(disease, final_risk, request.symptom_names, precautions)
    
    return {
        "disease": disease,
        "risk_score": final_risk,
        "risk_level": risk_label,
        "air_quality": aq_data,
        "advisory": advisory,
        "symptom_match_details": pred_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
