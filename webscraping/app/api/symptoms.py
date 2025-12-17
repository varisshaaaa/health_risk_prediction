from fastapi import APIRouter, Depends
from app.services.symptom_cleaner import extract_symptoms
from app.services.symptom_verifier import verify_symptom
from app.services.disease_scraper import scrape_disease_and_precautions
from app.services.retrain_service import trigger_retrain
from app.db.crud_symptoms import add_symptom
from app.db.crud_precautions import precautions_exist, insert_precautions
from app.db.database import get_db

router = APIRouter()

@router.post("/analyze-symptoms")
def analyze_symptoms(payload: dict, db=Depends(get_db)):

    checked = payload["checked"]
    free_text = payload["free_text"]
    known_symptoms = payload["known_symptoms"]

    final_symptoms, new_symptoms = extract_symptoms(
        checked, free_text, known_symptoms
    )

    # Step-1: handle NEW symptoms
    for s in new_symptoms:
        if verify_symptom(s):
            add_symptom(db, s)

    trigger_retrain(new_symptoms)

    # Step-2: precautions logic (known + new)
    for s in final_symptoms + new_symptoms:
        if not precautions_exist(db, s):
            scraped = scrape_disease_and_precautions(s)
            for d in scraped["diseases"]:
                insert_precautions(db, s, d, scraped["precautions"])

    return {
        "final_symptoms": final_symptoms,
        "new_symptoms": new_symptoms,
        "status": "processed"
    }
