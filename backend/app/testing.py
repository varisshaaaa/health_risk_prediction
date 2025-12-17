import pandas as pd
import json
import os

# Absolute paths relative to this script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # backend/
CSV_PATH = os.path.join(ROOT_DIR, "symptoms_and_disease.csv")

# Make sure the data folder exists for JSON
DATA_FOLDER = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)
JSON_PATH = os.path.join(DATA_FOLDER, "symptoms_master.json")

# Read CSV
df = pd.read_csv(CSV_PATH)

# Get all symptom columns (exclude 'disease')
symptoms = [c for c in df.columns if c != "disease"]

# Create master symptom registry
symptom_registry = {
    "confirmed_symptoms": symptoms,
    "pending_symptoms": {},   # new symptoms typed by users
    "confirmation_threshold": 3
}

# Save JSON
with open(JSON_PATH, "w") as f:
    json.dump(symptom_registry, f, indent=2)

print(f"✅ Master symptom registry created at {JSON_PATH}")
