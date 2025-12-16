import pandas as pd
from catboost import CatBoostClassifier
import os

def train_model():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'symptoms_and_disease.csv') # Assuming it's in root or check path
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_model.cbm')

    # Create model directory if not exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load Data from CSV
    print(f"Loading base data from {DATA_PATH}...")
    df_base = pd.read_csv(DATA_PATH)
    
    # Load New Data from DB (Postgres/SQLite)
    # Using SQLAlchemy engine from backend code (needs imports)
    try:
        from backend.database import engine
        from backend.models import PredictionLog
        
        # Query: Get confirmed disease logs (assuming we treat predicted as ground truth for self-training OR we only use corrected ones)
        # For now, we'll use the 'PredictionLog' as a source of new vectors.
        # Ideally, we should have a 'LabelledLog' where user confirmed the disease.
        # Current instruction: "checks if al teh symtoms and disease is alraey in thedatset"
        # We will fetch logs where we have symptoms and a disease label.
        
        query = "SELECT symptoms_vector, predicted_disease FROM feature_store"
        df_new_raw = pd.read_sql(query, engine)
        
        if not df_new_raw.empty:
            print(f"Found {len(df_new_raw)} new records in DB.")
            
            # Convert JSON vector back to columns
            # df_base columns: [Symptom1, Symptom2, ..., Disease]
            # df_new_raw columns: [symptoms_vector (list), predicted_disease]
            
            # 1. Get feature names from base df (excluding Disease)
            feature_names = [c for c in df_base.columns if c != 'Disease']
            
            # 2. Parse symptoms_vector
            # Ensure it fits the same schema
            new_rows = []
            for _, row in df_new_raw.iterrows():
                vec = row['symptoms_vector']
                disease = row['predicted_disease']
                if isinstance(vec, list) and len(vec) == len(feature_names):
                    record = dict(zip(feature_names, vec))
                    record['Disease'] = disease
                    new_rows.append(record)
            
            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df = pd.concat([df_base, df_new], ignore_index=True)
                print(f"Merged dataset size: {len(df)}")
            else:
                df = df_base
        else:
            df = df_base
            
    except Exception as e:
        print(f"Warning: Could not load new data from DB: {e}")
        df = df_base

    X = df.drop('Disease', axis=1)
    y = df['Disease']

    # Train Model
    print("Training CatBoost Classifier...")
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=False)
    model.fit(X, y)

    # Save Model
    print(f"Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    print("Model training completed successfully.")

if __name__ == "__main__":
    train_model()
