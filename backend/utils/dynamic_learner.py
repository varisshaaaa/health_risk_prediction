import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Persistency Configuration for Railway / Docker
# If DATA_DIR env var is set (e.g., /app/data), we use that for mutable files (CSVs, PKLs)
# Otherwise we default to local project paths.
DATA_DIR = os.getenv("DATA_DIR", PROJECT_ROOT) # Defaults to project root if not set
BACKEND_DATA_DIR = os.getenv("DATA_DIR", BASE_DIR) # Defaults to backend/ if not set (for model)

# Input Data (Source of Truth)
# We might copy the initial seeds to the persistent dir if they don't exist there yet.
SEED_DATA_PATH = os.path.join(PROJECT_ROOT, "symptoms_and_disease.csv")
SEED_PRECAUTIONS_PATH = os.path.join(PROJECT_ROOT, "backend", "precautions.csv")

# Active Data Paths (Mutable)
DATA_PATH = os.path.join(DATA_DIR, "symptoms_and_disease.csv")
PRECAUTIONS_PATH = os.path.join(DATA_DIR, "precautions.csv")
MODEL_PATH = os.path.join(BACKEND_DATA_DIR, "disease_model.pkl")
ENCODER_PATH = os.path.join(BACKEND_DATA_DIR, "label_encoder.pkl")
# Webscraping Service URL
# If running in Docker Compose, this might need to be the service name 'webscraping'
# For local run, localhost:8001
WEBSCRAPER_URL = os.getenv("WEBSCRAPER_URL", "http://localhost:8001")

def scrape_disease(symptom):
    """
    Calls the webscraping microservice to get diseases for a symptom.
    """
    print(f"REQUESTING SCRAPE for symptom: {symptom}")
    try:
        url = f"{WEBSCRAPER_URL}/scrape/disease"
        response = requests.post(url, json={"symptom": symptom}, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Scraper service returned {response.status_code}")
            return {"diseases": [], "precautions": []}
    except Exception as e:
        print(f"Failed to connect to webscraper: {e}")
        return {"diseases": [], "precautions": []}

def verify_symptom_with_service(symptom):
    """
    Calls webscraping service to verify if a symptom is real.
    """
    try:
        url = f"{WEBSCRAPER_URL}/verify/symptom"
        response = requests.post(url, json={"symptom": symptom}, timeout=10)
        if response.status_code == 200:
            return response.json().get("is_valid", False)
    except Exception as e:
        print(f"Verification failed: {e}")
    return False

def scrape_precautions_for_disease(disease_name):
    # This logic was requested to be separate: "If missing -> scrape diseases + precautions"
    # Actually current scraper returns both. 
    # But if we need JUST precautions for a known disease that has none:
    # We can reuse the scraper or just search for the disease name as a "symptom" keyword?
    # Or ideally, the scraper endpoint should support a generic query.
    # For now, we'll try to use the scrape_disease endpoint using the disease name, 
    # relying on the scraper to find relevant info.
    print(f"Fetching precautions for disease: {disease_name}")
    data = scrape_disease(disease_name) 
    return data.get("precautions", [])

def integrate_new_symptom(symptom):
    """
    Logic:
    User Input -> New Symptom
    1. Verify (web)
    2. Update DB (CSV)
    3. Retrain
    """
    
    # 1. Verify
    print(f"Verifying new symptom: {symptom}")
    is_valid = verify_symptom_with_service(symptom)
    if not is_valid:
        print(f"Symptom '{symptom}' could not be verified online. Skipping.")
        return False

    # Establish Mutable Data if not exists
    if not os.path.exists(DATA_PATH):
        if os.path.exists(SEED_DATA_PATH):
            print(f"Initializing mutable data from seed: {SEED_DATA_PATH} -> {DATA_PATH}")
            # We used open/write to copy
            import shutil
            shutil.copy(SEED_DATA_PATH, DATA_PATH)
        else:
            print("Error: Dataset not found in seed or mutable path.")
            return False

    df = pd.read_csv(DATA_PATH)
    
    # 2. Scrape Data (Get Diseases)
    data = scrape_disease(symptom)
    diseases = data.get('diseases', [])
    precautions = data.get('precautions', [])

    if not diseases:
        print(f"No diseases found for {symptom}. Cannot integrate.")
        return False

    # 3. Update DataFrame (symptoms_and_disease.csv)
    # Add symptom column if not exists
    if symptom not in df.columns:
        df[symptom] = 0
        print(f"Added new column: {symptom}")

    for disease in diseases:
        # User Logic: "New symptoms retrain once, not spam"
        # We add the new connection.
        
        # Add new disease row if not exists
        if disease not in df['Disease'].values:
            new_row = {col: 0 for col in df.columns}
            new_row['Disease'] = disease
            new_row[symptom] = 1 # Strong link
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Added new disease row: {disease}")
            
            # Also add precautions for this new disease immediately
            update_precautions_db_entry(disease, symptom, precautions)
        else:
            # Mark existing disease's symptom as 1
            df.loc[df['Disease'] == disease, symptom] = 1
            print(f"Updated existing disease '{disease}' with symptom '{symptom}'")

    # Save to CSV
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset updated with symptom '{symptom}'")

    # 4. Retrain Model
    train_model(df)
    
    return True

def update_precautions_db_entry(disease, symptom, precautions_list):
    """
    Updates the precautions CSV.
    """
    if not precautions_list:
        return

    if os.path.exists(PRECAUTIONS_PATH):
        prec_df = pd.read_csv(PRECAUTIONS_PATH)
    else:
        # Check seed
        if os.path.exists(SEED_PRECAUTIONS_PATH):
             import shutil
             shutil.copy(SEED_PRECAUTIONS_PATH, PRECAUTIONS_PATH)
             prec_df = pd.read_csv(PRECAUTIONS_PATH)
        else:
             prec_df = pd.DataFrame(columns=["Disease", "Symptom", "Precaution"])
    
    for prec in precautions_list:
        # Check duplicates
        is_present = ((prec_df['Disease'] == disease) & 
                      (prec_df['Precaution'] == prec)).any()
        
        if not is_present:
            new_row = {"Disease": disease, "Symptom": symptom, "Precaution": prec}
            prec_df = pd.concat([prec_df, pd.DataFrame([new_row])], ignore_index=True)
    
    prec_df.to_csv(PRECAUTIONS_PATH, index=False)
    print(f"✅ Precautions dataset updated for {disease}")

def train_model(df=None):
    """
    Retrains the RandomForest model using the current dataset.
    """
    if df is None:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        elif os.path.exists(SEED_DATA_PATH):
            print("Loading from seed data for initial training...")
            df = pd.read_csv(SEED_DATA_PATH)
            # We don't necessarily save to DATA_PATH yet unless we modify it, 
            # but for consistency let's ensure DATA_PATH exists after first train? 
            # OR just wait for the first new symptom. 
            # Let's simple load.
        else:
            print("No data to train on.")
            return
        
    print("Training model...")
    # Clean data (fill NaNs)
    df = df.fillna(0)
    
    # X/y split
    # Assume 'Disease' is the target
    if 'Disease' not in df.columns:
        print("Dataset missing 'Disease' column.")
        return

    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    # Verify we have numeric data
    # Some columns might be non-numeric if something went wrong, force conversion
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train ensemble model (RandomForest as requested)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X, y_encoded)

    # Calculate basic accuracy on training set
    score = rf_model.score(X, y_encoded)
    
    model_data = {
        'model': rf_model,
        'encoder': label_encoder,
        'symptom_columns': list(X.columns),
        'accuracy': score
    }
    joblib.dump(model_data, MODEL_PATH)

    print(f"✅ Model retrained. Accuracy: {score:.4f}")

if __name__ == "__main__":
    train_model()
