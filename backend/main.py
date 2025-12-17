from fastapi import FastAPI
import uvicorn
import os
import sys

# Ensure backend package is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

# Database
from backend.database.database import engine, Base

# Initialize Tables
Base.metadata.create_all(bind=engine)

# API Router
from backend.api.health import router as health_router

app = FastAPI(title="Integrated Health Risk Predictor")

app.include_router(health_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Note: Use backend.main:app string for uvicorn workers
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
