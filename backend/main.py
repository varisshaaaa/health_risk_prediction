from fastapi import FastAPI
import uvicorn
import os
import sys
from contextlib import asynccontextmanager

# Ensure backend package is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

# Database
from backend.database.database import engine, Base, init_database, test_database_connection

# API Router
from backend.api.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database with proper testing and seeding
    print("=" * 50)
    print("🚀 Health Risk Predictor API Starting...")
    print("=" * 50)
    
    try:
        # Test and initialize database
        db_available = init_database()
        if db_available:
            print("✅ Database fully initialized with all tables and seed data")
        else:
            print("⚠️ Running without database - using CSV files only")
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")
        print("   Running in limited mode with CSV files only")
    
    print("=" * 50)
    print("✅ API Ready to accept requests")
    print("=" * 50)
    
    yield
    
    # Shutdown
    print("👋 Shutting down Health Risk Predictor API...")

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
