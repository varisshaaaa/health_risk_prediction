import sys
import os
import pandas as pd
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'symptoms_and_disease.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'ml_models', 'label_encoder.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'database', 'symptoms_and_disease.csv')
PRECAUTIONS_PATH = os.path.join(BASE_DIR, 'database', 'Precautions.csv')

class DiseaseRiskOrchestrator:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.df = None
        self.precautions_df = None
        self.load_resources()

    def load_resources(self):
        """Loads model, encoder, and dataset."""
        try:
            if os.path.exists(MODEL_PATH):
                loaded_data = joblib.load(MODEL_PATH)
                if isinstance(loaded_data, dict):
                    self.model = loaded_data.get('model')
                    self.encoder = loaded_data.get('encoder')
                else:
                    self.model = loaded_data
                    if os.path.exists(ENCODER_PATH):
                        self.encoder = joblib.load(ENCODER_PATH)
            else:
                print(f"Model not found at {MODEL_PATH}")

            if os.path.exists(DATA_PATH):
                self.df = pd.read_csv(DATA_PATH)
            else:
                print(f"Data not found at {DATA_PATH}")
            
            self.reload_precautions() 
            
        except Exception as e:
            print(f"Error loading resources: {e}")

    def reload_precautions(self):
        """Reloads precautions from CSV to pick up scraping updates."""
        try:
            if os.path.exists(PRECAUTIONS_PATH):
                self.precautions_df = pd.read_csv(PRECAUTIONS_PATH)
                # Validate format - ensure correct columns exist
                if not self.precautions_df.empty:
                    if 'Disease' not in self.precautions_df.columns or 'Precaution' not in self.precautions_df.columns:
                        print(f"Warning: Precautions CSV has incorrect format. Expected columns: Disease, Precaution, Source")
                        print(f"Found columns: {list(self.precautions_df.columns)}")
                        # Try to fix if old format
                        if 'disease' in self.precautions_df.columns and 'precautions' in self.precautions_df.columns:
                            # Old format detected - create empty DataFrame
                            self.precautions_df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])
                            print("Precautions CSV format corrected. Please re-scrape to populate data.")
                        else:
                            self.precautions_df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])
            else:
                self.precautions_df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])
        except Exception as e:
            print(f"Error loading precautions CSV: {e}")
            self.precautions_df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])

    def predict_diseases(self, symptoms_input, top_n=5):
        """
        Predicts disease based on symptoms input.
        symptoms_input: list of strings (cleaned symptoms)
        """
        if not self.model or self.df is None:
            return [{"disease": "System Initializing", "probability": 0, "severity": "Low", "matched_symptoms": []}]

        # Prepare input vector
        feature_cols = self.df.drop("Disease", axis=1).columns
        X_input = np.zeros(len(feature_cols))
        
        matched_symptoms_clean = []
        
        for i, col in enumerate(feature_cols):
            if col in symptoms_input:
                X_input[i] = 1
                matched_symptoms_clean.append(col)

        # Quick check: If no symptoms matched at all, don't ask the model (GIGO)
        if not matched_symptoms_clean:
             return [{"disease": "Healthy / No Data", "probability": 100, "severity": "Low", "matched_symptoms": []}]

        X_input = X_input.reshape(1, -1)
        
        try:
            # SAFETY CHECK: Model expected features vs Input features
            # Verify shape if possible.
            try:
                # Some sklearn models expose n_features_in_
                if hasattr(self.model, "n_features_in_"):
                    if self.model.n_features_in_ != X_input.shape[1]:
                        print(f"Feature Mismatch! Model expects {self.model.n_features_in_}, got {X_input.shape[1]}")
                        # Auto-retrain trigger could go here, but for now fallback.
                        # Attempt to pad/truncate? Risky.
                        return [{"disease": "System Syncing (Retrain Required)", "probability": 0, "severity": "Low", "matched_symptoms": []}]
            except:
                pass

            probs = self.model.predict_proba(X_input)[0]
        except Exception as e:
            print(f"Prediction Error: {e}")
            return [{"disease": "Unknown Condition", "probability": 0, "severity": "Low", "matched_symptoms": matched_symptoms_clean}]

        top_indices = np.argsort(probs)[-top_n:][::-1]
        results = []

        for idx in top_indices:
            disease = self.encoder.inverse_transform([idx])[0]
            probability = probs[idx] # 0-1
            
            # Filter low probability noise
            if probability < 0.05:
                continue

            # Severity logic
            try:
                disease_row = self.df[self.df['Disease'] == disease].iloc[0]
                total_symptoms = sum(disease_row[1:]) 
                
                matched_for_this_disease = [s for s in matched_symptoms_clean if disease_row[s] == 1]
                symptom_count = len(matched_for_this_disease)
                
                match_ratio = symptom_count / total_symptoms if total_symptoms else 0

                if match_ratio >= 0.8: severity = "High"
                elif match_ratio >= 0.5: severity = "Moderate"
                else: severity = "Low"
                    
                results.append({
                    "disease": disease,
                    "probability": round(probability * 100, 2), 
                    "severity": severity,
                    "matched_symptoms": matched_for_this_disease
                })
            except Exception as e:
                print(f"Severity Calc Error for {disease}: {e}")
                continue
        
        # Fallback if no disease met expectation
        if not results:
             return [{"disease": "Unknown Condition", "probability": 0, "severity": "Low", "matched_symptoms": matched_symptoms_clean}]

        return results

    def get_precautions(self, disease):
        """
        Fetches precautions from both CSV and database.
        Combines results from both sources with deduplication.
        """
        self.reload_precautions()
        
        precautions = []
        
        # Load from CSV
        if not self.precautions_df.empty:
            # Handle different possible column names
            if 'Disease' in self.precautions_df.columns and 'Precaution' in self.precautions_df.columns:
                rows = self.precautions_df[self.precautions_df['Disease'] == disease]
                if not rows.empty:
                    csv_precautions = rows['Precaution'].unique().tolist()
                    precautions.extend([p for p in csv_precautions if p and str(p).strip()])
        
        # Also load from database
        try:
            from backend.database.database import SessionLocal
            from backend.database.models import Precaution
            
            db = SessionLocal()
            db_precautions = db.query(Precaution).filter(Precaution.disease == disease).all()
            db_precautions_list = [p.content for p in db_precautions if p.content and str(p.content).strip()]
            precautions.extend(db_precautions_list)
            db.close()
        except Exception as e:
            # Fallback if database access fails
            import sys
            if 'backend.utils.logger' in sys.modules:
                from backend.utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Failed to load precautions from DB for {disease}: {e}")
            else:
                print(f"Warning: Failed to load precautions from DB for {disease}: {e}")
        
        # Deduplicate while preserving order
        seen = set()
        unique_precautions = []
        for p in precautions:
            if not p:
                continue
            p_str = str(p).strip()
            p_lower = p_str.lower()
            if p_lower and p_lower not in seen:
                unique_precautions.append(p_str)
                seen.add(p_lower)
        
        return unique_precautions
