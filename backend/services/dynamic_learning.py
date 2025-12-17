"""
Dynamic Learning Module
Handles new symptom detection, verification, web scraping, and model retraining.
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_PATH = os.path.join(BASE_DIR, 'database', 'symptoms_and_disease.csv')
PRECAUTIONS_PATH = os.path.join(BASE_DIR, 'database', 'Precautions.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'symptoms_and_disease.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'ml_models', 'label_encoder.pkl')

# Ensure backend package is in path
sys.path.append(os.path.dirname(BASE_DIR))

# Import logger
try:
    from backend.utils.logger import get_logger, log_dynamic_learning
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    log_dynamic_learning = None

# Import Database components
try:
    from backend.database.database import SessionLocal
    from backend.database.models import Precaution, SymptomLog, TrainingData
except ImportError:
    logger.warning("Could not import Database components. SQL updates will fail.")
    SessionLocal = None
    TrainingData = None

# Import web scraper
from backend.services.web_scraper import scrape_disease_and_precautions, verify_symptom as verify_symptom_logic


def scrape_disease(symptom):
    """
    Calls the local webscraping service to get diseases for a symptom.
    """
    logger.info(f"Scraping for symptom: {symptom}")
    try:
        return scrape_disease_and_precautions(symptom)
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
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
        logger.warning("SQL Session missing, fallback to CSV.")
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH)
        return pd.DataFrame()

    db = SessionLocal()
    try:
        rows = db.query(TrainingData).all()
        
        if not rows:
            # If SQL empty, try seeding from CSV if exists
            if os.path.exists(DATA_PATH):
                logger.info("Seeding SQL from CSV...")
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
        
        df = pd.DataFrame(data_list).fillna(0)
        logger.info(f"Loaded {len(df)} records from SQL")
        return df
        
    except Exception as e:
        logger.error(f"Error loading from SQL: {e}")
        logger.info("Fallback: Loading from CSV...")
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
    if not SessionLocal:
        return False
    
    db = SessionLocal()
    try:
        record = db.query(TrainingData).filter(TrainingData.disease == disease).first()
        if record:
            # Update existing
            profile = dict(record.symptom_profile)
            profile[symptom] = 1  # Strong link
            record.symptom_profile = profile
            record.updated_at = datetime.utcnow()
        else:
            # Create new record
            profile = {symptom: 1}
            db.add(TrainingData(disease=disease, symptom_profile=profile))
        
        db.commit()
        logger.info(f"SQL updated: disease='{disease}', symptom='{symptom}'")
        return True
        
    except Exception as e:
        logger.error(f"SQL Save Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def update_precautions_db_entry(disease, symptom, precautions):
    """
    Adds precautions to the database for a disease.
    
    Args:
        disease: Disease name
        symptom: Symptom that triggered the scraping
        precautions: List of precaution strings
        
    Returns:
        bool: Success status
    """
    if not SessionLocal:
        logger.warning("Database session not available. Cannot update precautions in DB.")
        return False
    
    if not precautions:
        logger.warning(f"No precautions provided for {disease}")
        return False
    
    if not disease or not isinstance(disease, str):
        logger.error(f"Invalid disease name: {disease}")
        return False
    
    db = SessionLocal()
    added_count = 0
    try:
        for precaution_text in precautions:
            if not precaution_text or not isinstance(precaution_text, str):
                continue
                
            precaution_clean = precaution_text.strip()
            if len(precaution_clean) < 10:
                continue
            
            # Check if already exists
            exists = db.query(Precaution).filter(
                Precaution.disease == disease,
                Precaution.content == precaution_clean
            ).first()
            
            if not exists:
                p = Precaution(
                    disease=disease,
                    content=precaution_clean,
                    severity_level='BASIC',
                    source=f'web_scrape:{symptom}'
                )
                db.add(p)
                added_count += 1
        
        if added_count > 0:
            db.commit()
            logger.info(f"✅ Added {added_count} new precautions to database for {disease}")
            return True
        else:
            logger.info(f"No new precautions to add to database for {disease} (all already exist)")
            return True  # Still return True as this is not an error
        
    except Exception as e:
        logger.error(f"Precaution DB Error for {disease}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        return False
    finally:
        db.close()


def update_precautions_csv(disease: str, precautions: list) -> bool:
    """
    Updates Precautions.csv file with new precautions.
    Format: Disease,Precaution,Source
    
    Args:
        disease: Disease name
        precautions: List of precaution strings
        
    Returns:
        bool: Success status
    """
    try:
        # Read existing CSV or create new
        if os.path.exists(PRECAUTIONS_PATH):
            try:
                df = pd.read_csv(PRECAUTIONS_PATH)
                # Ensure correct column names - fix if wrong format
                if 'Disease' not in df.columns or 'Precaution' not in df.columns:
                    logger.warning(f"Precautions CSV has wrong format. Recreating with correct format.")
                    df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])
            except Exception as e:
                logger.warning(f"Error reading Precautions CSV: {e}. Creating new file.")
                df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])
        else:
            df = pd.DataFrame(columns=['Disease', 'Precaution', 'Source'])
        
        if not precautions:
            logger.warning(f"No precautions provided for {disease}")
            return False
        
        # Add new precautions
        new_rows = []
        for precaution in precautions:
            if not precaution or not isinstance(precaution, str):
                continue
                
            precaution_clean = precaution.strip()
            if len(precaution_clean) < 10:
                continue
                
            # Check if already exists
            exists = False
            if not df.empty:
                exists = not df[(df['Disease'] == disease) & (df['Precaution'] == precaution_clean)].empty
            
            if not exists:
                new_rows.append({
                    'Disease': disease,
                    'Precaution': precaution_clean,
                    'Source': 'web_scrape'
                })
        
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(PRECAUTIONS_PATH, index=False)
            logger.info(f"✅ Added {len(new_rows)} precautions to CSV for {disease}")
            return True
        else:
            logger.info(f"No new precautions to add for {disease} (all already exist)")
            return False
        
    except Exception as e:
        logger.error(f"CSV Precautions Update Error for {disease}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def update_csv_with_new_symptom(symptom, diseases):
    """
    Updates the CSV file with a new symptom column.
    """
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            
            if symptom not in df.columns:
                # Add new column with 0s
                df[symptom] = 0
                
                # Set 1 for affected diseases
                for disease in diseases:
                    df.loc[df['Disease'] == disease, symptom] = 1
                
                df.to_csv(DATA_PATH, index=False)
                logger.info(f"Added symptom '{symptom}' to CSV")
                return True
                
    except Exception as e:
        logger.error(f"CSV Update Error: {e}")
        
    return False


def integrate_new_symptom(symptom):
    """
    Main integration pipeline:
    1. Verify symptom is valid (web search)
    2. Scrape associated diseases and precautions
    3. Update SQL database
    4. Update CSV file
    5. Retrain model
    
    Returns:
        bool: Success status
    """
    logger.info(f"Starting integration for new symptom: {symptom}")
    
    # 1. Verify symptom is real
    is_valid = verify_symptom_with_service(symptom)
    if not is_valid:
        logger.warning(f"Symptom '{symptom}' could not be verified online. Skipping.")
        if log_dynamic_learning:
            log_dynamic_learning(logger, symptom, [], False)
        return False

    # 2. Scrape diseases and precautions
    data = scrape_disease(symptom)
    diseases = data.get('diseases', [])
    precautions = data.get('precautions', [])

    if not diseases:
        logger.warning(f"No diseases found for '{symptom}'. Cannot integrate.")
        if log_dynamic_learning:
            log_dynamic_learning(logger, symptom, [], False)
        return False

    logger.info(f"Found {len(diseases)} diseases and {len(precautions)} precautions for symptom '{symptom}'")

    # 3. Update SQL database and CSV files
    for disease in diseases:
        save_new_training_data_to_sql(disease, symptom)
        # Update precautions in database
        update_precautions_db_entry(disease, symptom, precautions)
        # Update precautions in CSV
        update_precautions_csv(disease, precautions)

    # 4. Update symptoms CSV file
    update_csv_with_new_symptom(symptom, diseases)

    # 5. Retrain model
    df = load_training_data_from_sql()
    if not df.empty:
        train_model(df)
    
    if log_dynamic_learning:
        log_dynamic_learning(logger, symptom, diseases, True)
    
    logger.info(f"✅ Successfully integrated symptom '{symptom}'")
    return True


def train_model(df=None):
    """
    Retrains the RandomForest model using data from SQL (or passed df).
    """
    # Try to use the centralized ml_model training
    try:
        from backend.ml_models.ml_model import train_symptom_disease_model, save_model
        
        if df is None:
            df = load_training_data_from_sql()
            
        if df.empty:
            logger.warning("No data to train on.")
            return
        
        model_data = train_symptom_disease_model(df)
        if model_data:
            save_model(model_data)
            logger.info(f"✅ Model retrained. Accuracy: {model_data['accuracy']:.4f}")
        return
        
    except ImportError:
        pass
    
    # Fallback: local training
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    
    if df is None:
        df = load_training_data_from_sql()
        
    if df.empty:
        logger.warning("No data to train on.")
        return
        
    logger.info(f"Training model on {len(df)} records...")
    df = df.fillna(0)
    
    if 'Disease' not in df.columns:
        logger.error("Dataset missing 'Disease' column.")
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

    logger.info(f"✅ Model retrained. Accuracy: {score:.4f}")


if __name__ == "__main__":
    train_model()
