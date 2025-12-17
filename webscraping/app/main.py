from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

# Import services
from .services.disease_scraper import scrape_disease_and_precautions
from .services.symptom_verifier import verify_symptom
from .db.database import init_db

app = FastAPI(title="WebScraping Microservice")

class ScrapeRequest(BaseModel):
    symptom: str

class VerifyRequest(BaseModel):
    symptom: str

class DisesePrecautionRequest(BaseModel):
    disease: str

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "webscraping"}

@app.post("/verify/symptom")
def verify_symptom_endpoint(request: VerifyRequest):
    """
    Verifies if a symptom exists/is valid via web search.
    """
    is_valid = verify_symptom(request.symptom)
    return {"symptom": request.symptom, "is_valid": is_valid}

@app.post("/scrape/disease")
def scrape_disease_endpoint(request: ScrapeRequest):
    """
    Scrapes diseases and precautions associated with a symptom.
    """
    data = scrape_disease_and_precautions(request.symptom)
    # data format: {"diseases": [...], "precautions": [...]}
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
