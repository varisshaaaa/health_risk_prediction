from fastapi import FastAPI
import uvicorn
import os
import sys
from contextlib import asynccontextmanager

# Ensure backend package is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

# Database
from backend.database.database import engine, Base

# API Router
from backend.api.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Tables
    try:
        print("Initializing Database Tables...")
        Base.metadata.create_all(bind=engine)
        print("Database Tables Initialized.")
    except Exception as e:
        print(f"WARNING: Database initialization failed. Functionality may be limited. Error: {e}")
    yield
    # Shutdown logic (if any)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Integrated Health Risk Predictor", lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for Railway internal comms
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Health Risk Prediction API V4.1 is Online", "docs": "/docs"}

app.include_router(health_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Note: Use backend.main:app string for uvicorn workers
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
