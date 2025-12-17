import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/
DATA_PATH = os.path.join(BASE_DIR, 'database', 'symptoms_and_disease.csv')
PRECAUTIONS_PATH = os.path.join(BASE_DIR, 'database', 'Precautions.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'symptoms_and_disease.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'ml_models', 'label_encoder.pkl')

WEBSCRAPER_URL = os.getenv("WEBSCRAPER_URL", "http://localhost:8001")

# Import Database components
# Ensure backend package is in path or installed
sys.path.append(os.path.dirname(BASE_DIR)) # Add project root to path

try:
    from backend.database.database import SessionLocal, engine
    from backend.database.models import Precaution, SymptomLog
    from sqlalchemy import text
except ImportError:
    print("Warning: Could not import Database components. SQL updates will fail.")
    SessionLocal = None

from backend.services.web_scraper import scrape_disease_and_precautions, verify_symptom as verify_symptom_logic

def scrape_disease(symptom):
    """
    Calls the local webscraping service to get diseases for a symptom.
    """
    print(f"Scraping for symptom: {symptom}")
    try:
        return scrape_disease_and_precautions(symptom)
    except Exception as e:
        print(f"Scraping failed: {e}")
        return {"diseases": [], "precautions": []}

def verify_symptom_with_service(symptom):
    """
    Calls webscraping service to verify if a symptom is real.
    """
    return verify_symptom_logic(symptom)

def load_training_data_from_sql():
    """
    Loads training data from PostgreSQL into a DataFrame.
    Table: TrainingData (disease, symptom_profile=JSON)
    """
    if not SessionLocal:
        print("SQL Session missing, fallback to CSV.")
        if os.path.exists(DATA_PATH): return pd.read_csv(DATA_PATH)
        return pd.DataFrame()

    db = SessionLocal()
    try:
        from backend.database.models import TrainingData
        rows = db.query(TrainingData).all()
        
        if not rows:
            # If SQL empty, try seeding from CSV if exists
            if os.path.exists(DATA_PATH):
                print("Seeding SQL from CSV...")
                seed_df = pd.read_csv(DATA_PATH)
                for _, row in seed_df.iterrows():
                    disease = row['Disease']
                    profile = row.drop('Disease').to_dict()
                    db.add(TrainingData(disease=disease, symptom_profile=profile))
                db.commit()
                return seed_df
            return pd.DataFrame()

        # Convert SQL rows to DataFrame
        data_list = []
        for r in rows:
            item = r.symptom_profile.copy()
            item['Disease'] = r.disease
            data_list.append(item)
        
        return pd.DataFrame(data_list).fillna(0)
    except Exception as e:
        print(f"Error loading from SQL: {e}")
        print("Fallback: Loading from CSV...")
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH)
        return pd.DataFrame()
    finally:
        db.close()

def save_new_training_data_to_sql(disease, symptom, profile_update=None):
    """
    Updates or creates a training record in SQL.
    symptom: The new symptom being added (to ensure it exists in profile).
    profile_update: Full profile if available.
    """
    if not SessionLocal: return False
    
    db = SessionLocal()
    try:
        from backend.database.models import TrainingData
        
        # 1. Update ALL rows to include this new symptom with 0 if not present
        # actually, JSON is flexible, we might not need to explicit update all if we handle NaNs on load.
        # But for 'hot' vector consistency, let's logic it out.
        
        record = db.query(TrainingData).filter(TrainingData.disease == disease).first()
        if record:
            # Update existing
            profile = dict(record.symptom_profile)
            profile[symptom] = 1 # Strong link
            record.symptom_profile = profile
            record.updated_at = datetime.utcnow()
        else:
            # Create new
            # We need a base profile (all other symptoms 0? No, JSON handles sparsity)
            # We just save what we know. The DataFrame loader fills NaNs with 0.
            profile = {symptom: 1}
            db.add(TrainingData(disease=disease, symptom_profile=profile))
        
        db.commit()
        return True
    except Exception as e:
        print(f"SQL Save Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def integrate_new_symptom(symptom):
    """
    Logic:
    User Input -> New Symptom
    1. Verify (web)
    2. Update SQL
    3. Retrain
    """
    
    # 1. Verify
    print(f"Verifying new symptom: {symptom}")
    is_valid = verify_symptom_with_service(symptom)
    if not is_valid:
        print(f"Symptom '{symptom}' could not be verified online. Skipping.")
        return False

    # 2. Scrape Data (Get Diseases)
    data = scrape_disease(symptom)
    diseases = data.get('diseases', [])
    precautions = data.get('precautions', [])

    if not diseases:
        print(f"No diseases found for {symptom}. Cannot integrate.")
        return False

    # 3. Update Data in SQL
    for disease in diseases:
        save_new_training_data_to_sql(disease, symptom)
        # Also add precautions
        update_precautions_db_entry(disease, symptom, precautions)
        print(f"Updated SQL for disease '{disease}' with symptom '{symptom}'")

    # 4. Retrain Model
    df = load_training_data_from_sql()
    if not df.empty:
        train_model(df)
    
    return True

# ... update_precautions_db_entry ... (Keep as is, but ensures SQL used)

def train_model(df=None):
    """
    Retrains the RandomForest model using data from SQL (or passed df).
    """
    if df is None:
        df = load_training_data_from_sql()
        
    if df.empty:
        print("No data to train on.")
        return
        
    print(f"Training model on {len(df)} records...")
    df = df.fillna(0)
    
    if 'Disease' not in df.columns:
        print("Dataset missing 'Disease' column.")
        return

    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X, y_encoded)

    score = rf_model.score(X, y_encoded)
    
    model_data = {
        'model': rf_model,
        'encoder': label_encoder,
        'symptom_columns': list(X.columns),
        'accuracy': score
    }
    # Ensure dir exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    joblib.dump(model_data, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"✅ Model retrained using SQL data. Accuracy: {score:.4f}")

if __name__ == "__main__":
    train_model()
