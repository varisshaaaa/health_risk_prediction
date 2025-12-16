from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from .database import Base

class PredictionLog(Base):
    """
    Feature Store / Prediction Log
    Stores the input features and the resulting prediction for monitoring and drift detection.
    """
    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Inputs (Features)
    age = Column(Integer)
    gender = Column(String)
    city = Column(String)
    symptoms_vector = Column(JSON) # Store as JSON array
    symptom_names = Column(JSON)   # Store names for readability
    
    # 50/30/20 Split Components
    symptom_risk = Column(Float)      # 50%
    demographic_risk = Column(Float)  # 30%
    air_quality_risk = Column(Float)  # 20%
    
    # Outputs (Predictions)
    predicted_disease = Column(String)
    risk_score = Column(Float)
    risk_level = Column(String)

class SymptomLog(Base):
    """
    Log for new/unrecognized symptoms for dynamic learning.
    """
    __tablename__ = "symptom_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symptom_text = Column(String)
    count = Column(Integer, default=1)

class Precaution(Base):
    """
    Scraped precautions data.
    """
    __tablename__ = "precautions"
    
    id = Column(Integer, primary_key=True, index=True)
    disease = Column(String, index=True)
    content = Column(String)
    severity_level = Column(String) # BASIC, MODERATE, IMPORTANT, URGENT
    source = Column(String)

