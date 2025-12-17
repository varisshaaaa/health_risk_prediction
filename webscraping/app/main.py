from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

# Absolute imports (Railway-safe)
from app.services.disease_scraper import scrape_disease_and_precautions
from app.services.symptom_verifier import verify_symptom

app = FastAPI(title="WebScraping Microservice")


# ----------- Schemas -----------

class ScrapeRequest(BaseModel):
    symptom: str

class VerifyRequest(BaseModel):
    symptom: str


# ----------- Routes -----------

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "webscraping"}


@app.post("/verify/symptom")
def verify_symptom_endpoint(request: VerifyRequest):
    is_valid = verify_symptom(request.symptom)
    return {
        "symptom": request.symptom,
        "is_valid": is_valid
    }


@app.post("/scrape/disease")
def scrape_disease_endpoint(request: ScrapeRequest):
    """
    Scrapes diseases + precautions for a symptom.
    """
    data = scrape_disease_and_precautions(request.symptom)
    return {
        "symptom": request.symptom,
        "diseases": data.get("diseases", []),
        "precautions": data.get("precautions", [])
    }


# ----------- Entry Point -----------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Railway-safe
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
