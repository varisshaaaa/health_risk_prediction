import pandas as pd
import json

df = pd.read_csv("data/clean_symptoms.csv")

symptoms = [c for c in df.columns if c != "disease"]

symptom_registry = {
    "confirmed_symptoms": symptoms,
    "pending_symptoms": {},   # new symptoms typed by users
    "confirmation_threshold": 3
}

with open("data/symptoms_master.json", "w") as f:
    json.dump(symptom_registry, f, indent=2)

print("✅ Master symptom registry created")
