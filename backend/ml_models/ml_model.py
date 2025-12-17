"""
ML Model Training Module
Handles retraining of the symptom-disease prediction model.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'symptoms_and_disease.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'ml_models', 'label_encoder.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'database', 'symptoms_and_disease.csv')

# Import logger
try:
    from backend.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def load_training_data(source='csv'):
    """
    Loads training data from CSV or SQL.
    
    Args:
        source: 'csv' or 'sql'
    
    Returns:
        DataFrame with Disease and symptom columns
    """
    if source == 'csv':
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            logger.info(f"Loaded {len(df)} records from CSV")
            return df
        else:
            logger.error(f"CSV not found at {DATA_PATH}")
            return pd.DataFrame()
    
    elif source == 'sql':
        try:
            from backend.database.database import SessionLocal
            from backend.database.models import TrainingData
            
            db = SessionLocal()
            rows = db.query(TrainingData).all()
            db.close()
            
            if not rows:
                logger.warning("No data in SQL, falling back to CSV")
                return load_training_data('csv')
            
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
            logger.error(f"SQL load failed: {e}, falling back to CSV")
            return load_training_data('csv')
    
    return pd.DataFrame()


def train_symptom_disease_model(df=None, source='sql'):
    """
    Trains/Retrains the RandomForest model for disease prediction.
    
    Args:
        df: Optional DataFrame. If None, loads from source.
        source: 'csv' or 'sql' - where to load data from
    
    Returns:
        dict with model, encoder, accuracy, and symptom columns
    """
    if df is None:
        df = load_training_data(source)
    
    if df.empty:
        logger.error("No data available for training")
        return None
    
    # Clean data
    df = df.fillna(0)
    
    if 'Disease' not in df.columns:
        logger.error("Dataset missing 'Disease' column")
        return None
    
    # Prepare features and target
    X = df.drop("Disease", axis=1)
    y = df["Disease"]
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model trained with accuracy: {accuracy:.4f}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Number of diseases: {len(label_encoder.classes_)}")
    
    # Package model data
    model_data = {
        'model': model,
        'encoder': label_encoder,
        'symptom_columns': list(X.columns),
        'accuracy': accuracy,
        'n_features': X.shape[1],
        'n_classes': len(label_encoder.classes_)
    }
    
    return model_data


def save_model(model_data, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
    """
    Saves the trained model and encoder to disk.
    
    Args:
        model_data: dict containing model, encoder, and metadata
        model_path: path to save the model
        encoder_path: path to save the encoder
    """
    if model_data is None:
        logger.error("No model data to save")
        return False
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model package
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save encoder separately for compatibility
        joblib.dump(model_data['encoder'], encoder_path)
        logger.info(f"Encoder saved to {encoder_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False


def retrain_and_save(source='sql'):
    """
    Full retraining pipeline: load data, train, save.
    
    Args:
        source: 'csv' or 'sql'
    
    Returns:
        bool indicating success
    """
    logger.info(f"Starting retraining from {source}...")
    
    model_data = train_symptom_disease_model(source=source)
    
    if model_data:
        success = save_model(model_data)
        if success:
            logger.info(f"✅ Retraining complete. Accuracy: {model_data['accuracy']:.4f}")
        return success
    
    return False


def add_new_symptom_column(symptom_name, diseases_affected):
    """
    Adds a new symptom column to the training data.
    
    Args:
        symptom_name: name of the new symptom
        diseases_affected: list of diseases that have this symptom
    
    Returns:
        bool indicating success
    """
    try:
        # Update CSV
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            
            if symptom_name not in df.columns:
                # Add new column with 0s
                df[symptom_name] = 0
                
                # Set 1 for affected diseases
                for disease in diseases_affected:
                    df.loc[df['Disease'] == disease, symptom_name] = 1
                
                df.to_csv(DATA_PATH, index=False)
                logger.info(f"Added symptom '{symptom_name}' to CSV")
        
        # Update SQL
        try:
            from backend.database.database import SessionLocal
            from backend.database.models import TrainingData
            
            db = SessionLocal()
            
            for disease in diseases_affected:
                record = db.query(TrainingData).filter(
                    TrainingData.disease == disease
                ).first()
                
                if record:
                    profile = dict(record.symptom_profile)
                    profile[symptom_name] = 1
                    record.symptom_profile = profile
                else:
                    # Create new record
                    db.add(TrainingData(
                        disease=disease,
                        symptom_profile={symptom_name: 1}
                    ))
            
            db.commit()
            db.close()
            logger.info(f"Added symptom '{symptom_name}' to SQL")
            
        except Exception as e:
            logger.warning(f"SQL update failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add symptom column: {e}")
        return False


if __name__ == "__main__":
    # Manual retraining trigger
    print("Starting manual model retraining...")
    success = retrain_and_save(source='csv')
    if success:
        print("✅ Model retrained successfully!")
    else:
        print("❌ Retraining failed.")

