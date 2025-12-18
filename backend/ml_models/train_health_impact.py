"""
Train Health Impact Predictor Model
===================================

This script trains a Gradient Boosting model to predict health risk from air quality data.

Model: GradientBoostingRegressor
Purpose: Predict health risk score (0-1) from AQI and pollutant data
Features: AQI, PM2.5, PM10, NO2, CO, O3, SO2
Output: Health risk score (0-1)

This model is saved as: backend/ml_models/health_impact_predictor.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'health_impact_predictor.pkl')

def generate_training_data():
    """
    Generates synthetic training data for air quality health impact.
    
    In production, you would collect real data from:
    - Historical AQI data
    - Health impact studies
    - Medical records correlated with air quality
    
    For now, we generate realistic synthetic data based on known relationships:
    - Higher AQI = Higher health risk
    - Higher PM2.5/PM10 = Higher risk
    - Higher NO2, CO, SO2 = Higher risk
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic AQI values (1-5)
    aqi = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    
    # Generate pollutant concentrations based on AQI
    data = []
    for aq in aqi:
        if aq == 1:  # Good
            pm25 = np.random.uniform(0, 12)
            pm10 = np.random.uniform(0, 20)
            no2 = np.random.uniform(0, 40)
            co = np.random.uniform(0, 1)
            o3 = np.random.uniform(0, 50)
            so2 = np.random.uniform(0, 20)
            risk = np.random.uniform(0.0, 0.2)
        elif aq == 2:  # Fair
            pm25 = np.random.uniform(12, 35)
            pm10 = np.random.uniform(20, 50)
            no2 = np.random.uniform(40, 70)
            co = np.random.uniform(1, 2)
            o3 = np.random.uniform(50, 100)
            so2 = np.random.uniform(20, 50)
            risk = np.random.uniform(0.2, 0.4)
        elif aq == 3:  # Moderate
            pm25 = np.random.uniform(35, 55)
            pm10 = np.random.uniform(50, 100)
            no2 = np.random.uniform(70, 120)
            co = np.random.uniform(2, 4)
            o3 = np.random.uniform(100, 168)
            so2 = np.random.uniform(50, 100)
            risk = np.random.uniform(0.4, 0.6)
        elif aq == 4:  # Poor
            pm25 = np.random.uniform(55, 150)
            pm10 = np.random.uniform(100, 250)
            no2 = np.random.uniform(120, 200)
            co = np.random.uniform(4, 10)
            o3 = np.random.uniform(168, 208)
            so2 = np.random.uniform(100, 200)
            risk = np.random.uniform(0.6, 0.8)
        else:  # Very Poor (aq == 5)
            pm25 = np.random.uniform(150, 300)
            pm10 = np.random.uniform(250, 500)
            no2 = np.random.uniform(200, 400)
            co = np.random.uniform(10, 20)
            o3 = np.random.uniform(208, 400)
            so2 = np.random.uniform(200, 500)
            risk = np.random.uniform(0.8, 1.0)
        
        data.append({
            'AQI': aq,
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'CO': co,
            'O3': o3,
            'SO2': so2,
            'risk_score': risk
        })
    
    return pd.DataFrame(data)

def train_health_impact_model():
    """
    Trains Gradient Boosting model for health impact prediction.
    
    Returns:
        Trained model and evaluation metrics
    """
    print("=" * 60)
    print("Training Health Impact Predictor (Gradient Boosting)")
    print("=" * 60)
    
    # Generate or load training data
    print("Loading training data...")
    df = generate_training_data()
    print(f"Loaded {len(df)} samples")
    
    # Prepare features and target
    feature_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    X = df[feature_cols]
    y = df['risk_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Gradient Boosting Regressor
    print("\nTraining Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,      # Number of boosting stages
        learning_rate=0.1,     # Learning rate (shrinkage)
        max_depth=5,            # Maximum depth of trees
        min_samples_split=2,    # Minimum samples to split
        min_samples_leaf=1,     # Minimum samples in leaf
        subsample=0.8,          # Fraction of samples for each tree
        random_state=42,
        loss='squared_error'    # Loss function
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\n" + "=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print("=" * 60)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    
    return {
        'model': model,
        'model_type': 'GradientBoostingRegressor',
        'features': feature_cols,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

if __name__ == "__main__":
    results = train_health_impact_model()
    print("\n✅ Training complete!")
    print(f"Model: {results['model_type']}")
    print(f"Test R² Score: {results['test_r2']:.4f}")

