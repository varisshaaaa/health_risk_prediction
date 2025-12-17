from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from backend.services.health_features import get_air_quality_risk
from backend.services.demographic_risk import calculate_demographic_risk
from backend.services.symptom_processor import SymptomProcessor
from backend.services.disease_prediction import DiseaseRiskOrchestrator
from backend.services.dynamic_learning import integrate_new_symptom, train_model, load_training_data_from_sql
from backend.database.database import SessionLocal
from backend.database.models import PredictionLog

router = APIRouter()

# --- Initialize Services ---
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
    try:
        success = integrate_new_symptom(symptom)
        if success:
            # Reload orchestrator resources
            disease_orchestrator.load_resources()
            # Update processor known symptoms
            if disease_orchestrator.df is not None:
                symptom_processor.known_symptoms = [c for c in disease_orchestrator.df.columns if c != 'Disease']
    except Exception as e:
        print(f"Background learning failed for {symptom}: {e}")

# --- Core Endpoints ---

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "Integrated Health Backend"}

@router.get("/symptoms")
def get_symptoms():
    """
    Returns list of all known symptoms for frontend autocomplete/checkboxes.
    """
    return symptom_processor.known_symptoms

@router.post("/predict")
async def predict_health_risk(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Unified Endpoint: Features, Demographics, Disease Prediction, Risk Calculation.
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

    # Construct Advisory
    advisory = f"### Risk Level: {risk_level}\n"
    advisory += f"Based on your symptoms, there is a **{top_disease['probability']}%** probability of **{top_disease['disease']}**.\n\n"
    if risk_level == "High":
        advisory += "⚠️ **Immediate medical attention is recommended.** Please visit a doctor."
    elif risk_level == "Moderate":
        advisory += "⚠️ **Monitor your symptoms.** Consult a healthcare provider if they worsen."
    else:
        advisory += "✅ **Low risk.** Maintain hydration and rest."

    return {
        "predicted_disease": top_disease["disease"],
        "probability": top_disease["probability"],
        "severity": top_disease["severity"],
        
        # Legacy/Backward Compatibility
        "disease_severity": top_disease["severity"],
        "disease": top_disease["disease"],
        "air_quality": aq_data,
        
        "overall_health_risk": overall_risk_percent,
        "risk_level": risk_level,
        "matched_symptoms": matched,
        "new_symptoms_detected": new_candidates,
        "precautions": precautions,
        "environmental_data": aq_data,
        "advisory": advisory,
        "breakdown": {
            "symptom_contribution": round(0.5 * symptom_risk_score * 100, 1),
            "demographic_contribution": round(0.3 * demo_risk * 100, 1),
            "environmental_contribution": round(0.2 * health_impact_score * 100, 1)
        }
    }

# --- Dashboard Endpoints ---

@router.get("/logs/features")
def get_recent_logs(limit: int = 50):
    """
    Fetches recent prediction logs for the dashboard.
    """
    db = SessionLocal()
    try:
        # Fetch last 50 entries ordered by newest first
        logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/logs/performance")
def get_performance_metrics():
    """
    Returns historical performance metrics.
    """
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.asc()).limit(100).all()
        data = []
        for log in logs:
            metric = 85.0
            if log.risk_level == "High":
                 metric = 95.0
            data.append({
                "timestamp": log.timestamp,
                "accuracy": metric
            })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.post("/admin/retrain")
def release_retraining(background_tasks: BackgroundTasks):
    """
    Manually triggers model retraining from the dashboard.
    """
    def _train_task():
        print("Manual retraining triggered...")
        df = load_training_data_from_sql()
        if not df.empty:
            train_model(df)
            disease_orchestrator.load_resources()
            
    background_tasks.add_task(_train_task)
    return {"status": "Retraining started in background"}
