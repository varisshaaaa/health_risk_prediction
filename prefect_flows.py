"""
Prefect ML Workflow Orchestration
Health Risk Prediction System

This module implements a complete ML pipeline using Prefect for:
- Data ingestion and validation
- Feature engineering
- Model training with multiple algorithms
- Model evaluation and comparison
- Model versioning and deployment
"""

import os
import sys
import json
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

# ============================================================
# TASK 1: DATA INGESTION
# ============================================================
@task(name="Load Training Data", retries=3, retry_delay_seconds=10)
def load_data_task():
    """
    Loads training data from CSV or PostgreSQL database.
    Implements retry logic for database connection issues.
    """
    import pandas as pd
    
    data_path = os.path.join("backend", "database", "symptoms_and_disease.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from CSV")
        return df
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")


# ============================================================
# TASK 2: DATA VALIDATION
# ============================================================
@task(name="Validate Data Quality")
def validate_data_task(df):
    """
    Validates data quality and integrity.
    Checks for missing values, duplicates, and data types.
    """
    import pandas as pd
    
    validation_results = {
        "total_records": int(len(df)),
        "total_features": int(len(df.columns) - 1),  # Exclude 'Disease'
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "diseases_count": int(df['Disease'].nunique()) if 'Disease' in df.columns else 0,
        "validation_passed": True,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check for critical issues
    if validation_results["missing_values"] > len(df) * 0.1:
        validation_results["validation_passed"] = False
        validation_results["error"] = "Too many missing values (>10%)"
    
    if validation_results["diseases_count"] < 5:
        validation_results["validation_passed"] = False
        validation_results["error"] = "Insufficient disease classes"
    
    print(f"Data Validation Results: {json.dumps(validation_results, indent=2)}")
    return validation_results


# ============================================================
# TASK 3: FEATURE ENGINEERING
# ============================================================
@task(name="Feature Engineering")
def feature_engineering_task(df):
    """
    Performs feature engineering on the dataset.
    - Creates symptom severity scores
    - Generates symptom combinations
    """
    import pandas as pd
    import numpy as np
    
    # Get symptom columns
    symptom_cols = [col for col in df.columns if col != 'Disease']
    
    # Create total symptom count feature
    df['symptom_count'] = df[symptom_cols].sum(axis=1)
    
    # Create symptom severity indicator
    df['high_symptom_load'] = (df['symptom_count'] > df['symptom_count'].median()).astype(int)
    
    print(f"Feature engineering complete. Total features: {len(df.columns)}")
    
    return df, symptom_cols


# ============================================================
# TASK 4: MODEL TRAINING
# ============================================================
@task(name="Train Classification Model", retries=2)
def train_classification_model_task(df, symptom_cols):
    """
    Trains Random Forest classifier for disease prediction.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import joblib
    import pandas as pd
    
    # Prepare data
    X = df[symptom_cols].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    y = df['Disease']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Check if stratified split is possible (need at least 2 samples per class)
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_samples = min(class_counts.values())
    
    # Split data - use stratify only if all classes have >= 2 samples
    if min_samples >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        print(f"Warning: Some classes have only {min_samples} sample(s). Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results = {
        "model_type": "RandomForestClassifier",
        "accuracy": float(round(accuracy, 4)),
        "f1_score": float(round(f1, 4)),
        "n_classes": int(len(le.classes_)),
        "n_features": int(X.shape[1]),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    }
    
    print(f"Classification Model Results: {json.dumps(results, indent=2)}")
    
    # Save model
    model_data = {
        'model': model,
        'encoder': le,
        'symptom_columns': list(symptom_cols),
        'accuracy': float(accuracy)
    }
    
    model_path = os.path.join("backend", "ml_models", "symptoms_and_disease.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    
    return results, model_data


# ============================================================
# TASK 5: HEALTH IMPACT PREDICTOR (Gradient Boosting)
# ============================================================
@task(name="Train Health Impact Predictor")
def train_health_impact_predictor_task():
    """
    Trains Gradient Boosting model to predict health risk from air quality data.
    
    This model is saved as: backend/ml_models/health_impact_predictor.pkl
    Used in: backend/services/health_features.py
    
    Model: GradientBoostingRegressor
    Features: AQI, PM2.5, PM10, NO2, CO, O3, SO2
    Output: Health risk score (0-1)
    """
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from backend.ml_models.train_health_impact import train_health_impact_model
    
    print("Training Health Impact Predictor (Gradient Boosting)...")
    results = train_health_impact_model()
    
    results_summary = {
        "model_type": "GradientBoostingRegressor",
        "model_name": "health_impact_predictor",
        "purpose": "Predict health risk from air quality (AQI + pollutants)",
        "features": ["AQI", "PM2.5", "PM10", "NO2", "CO", "O3", "SO2"],
        "test_rmse": float(round(results['test_rmse'], 4)),
        "test_r2_score": float(round(results['test_r2'], 4)),
        "test_mae": float(round(results['test_mae'], 4)),
        "model_path": "backend/ml_models/health_impact_predictor.pkl"
    }
    
    print(f"Health Impact Predictor Results: {json.dumps(results_summary, indent=2)}")
    
    return results_summary


# ============================================================
# TASK 5B: RISK REGRESSION MODEL (Symptom-based)
# ============================================================
@task(name="Train Risk Regression Model")
def train_regression_model_task(df, symptom_cols):
    """
    Trains a regression model to predict health risk scores from symptoms.
    This is a separate model from the health impact predictor.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import pandas as pd
    
    # Create target variable (risk score based on symptom count)
    X = df[symptom_cols].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    y = df['symptom_count'] / df['symptom_count'].max()  # Normalize to 0-1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results = {
        "model_type": "GradientBoostingRegressor",
        "model_name": "symptom_risk_regressor",
        "purpose": "Predict risk score from symptom count",
        "rmse": float(round(rmse, 4)),
        "r2_score": float(round(r2, 4)),
        "training_samples": int(len(X_train))
    }
    
    print(f"Symptom Risk Regression Model Results: {json.dumps(results, indent=2)}")
    
    return results


# ============================================================
# TASK 6: CLUSTERING ANALYSIS
# ============================================================
@task(name="Symptom Clustering Analysis")
def clustering_analysis_task(df, symptom_cols):
    """
    Performs K-Means clustering on symptoms to identify patterns.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import pandas as pd
    
    X = df[symptom_cols].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    
    # Dimensionality reduction with PCA
    pca = PCA(n_components=min(10, len(symptom_cols)))
    X_pca = pca.fit_transform(X)
    
    # K-Means clustering
    n_clusters = min(5, len(df) // 10)
    if n_clusters < 2:
        n_clusters = 2
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    silhouette = silhouette_score(X_pca, clusters) if len(set(clusters)) > 1 else 0
    
    results = {
        "model_type": "KMeans + PCA",
        "n_clusters": int(n_clusters),
        "silhouette_score": float(round(silhouette, 4)),
        "pca_components": int(pca.n_components_),
        "variance_explained": float(round(sum(pca.explained_variance_ratio_), 4))
    }
    
    print(f"Clustering Results: {json.dumps(results, indent=2)}")
    
    return results


# ============================================================
# TASK 7: MODEL EVALUATION COMPARISON
# ============================================================
@task(name="Compare Model Performance")
def compare_models_task(classification_results, regression_results, clustering_results, health_impact_results=None):
    """
    Creates a comparison report of all ML experiments.
    """
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {
            "classification": classification_results,
            "health_impact_predictor": health_impact_results if health_impact_results else {},
            "symptom_risk_regression": regression_results,
            "clustering": clustering_results
        },
        "summary": {
            "best_classification_accuracy": classification_results["accuracy"],
            "health_impact_r2": health_impact_results.get("test_r2_score", "N/A") if health_impact_results else "N/A",
            "symptom_risk_r2": regression_results["r2_score"],
            "clustering_quality": clustering_results["silhouette_score"]
        }
    }
    
    # Save comparison report
    report_path = os.path.join("backend", "logs", "experiment_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Model Comparison Report saved to {report_path}")
    
    # Create Prefect artifact
    health_impact_section = ""
    if health_impact_results:
        health_impact_section = f"""
## Health Impact Predictor (Air Quality → Risk)
- **Model**: {health_impact_results.get('model_type', 'GradientBoostingRegressor')}
- **Purpose**: Predict health risk from AQI and pollutants
- **Test R² Score**: {health_impact_results.get('test_r2_score', 'N/A')}
- **Test RMSE**: {health_impact_results.get('test_rmse', 'N/A')}
- **Saved to**: {health_impact_results.get('model_path', 'N/A')}
"""
    
    markdown_report = f"""
# ML Experiment Results

## Classification (Disease Prediction)
- **Model**: {classification_results['model_type']}
- **Accuracy**: {classification_results['accuracy']}
- **F1 Score**: {classification_results['f1_score']}
{health_impact_section}
## Symptom Risk Regression
- **Model**: {regression_results['model_type']}
- **RMSE**: {regression_results['rmse']}
- **R² Score**: {regression_results['r2_score']}

## Clustering (Symptom Patterns)
- **Model**: {clustering_results['model_type']}
- **Silhouette Score**: {clustering_results['silhouette_score']}
- **Variance Explained**: {clustering_results['variance_explained']}

Generated at: {comparison['timestamp']}
"""
    
    create_markdown_artifact(
        key="ml-experiment-results",
        markdown=markdown_report,
        description="ML Model Comparison Results"
    )
    
    return comparison


# ============================================================
# TASK 8: NOTIFICATION
# ============================================================
@task(name="Send Notification")
def send_notification_task(comparison, success=True):
    """
    Sends notification on pipeline completion.
    """
    status = "SUCCESS" if success else "FAILED"
    
    timestamp = comparison.get('timestamp', datetime.now().isoformat())
    
    if success and 'summary' in comparison:
        message = f"""
    ========================================
    ML PIPELINE {status}
    ========================================
    Timestamp: {timestamp}
    
    Classification Accuracy: {comparison['summary'].get('best_classification_accuracy', 'N/A')}
    Health Impact R² Score: {comparison['summary'].get('health_impact_r2', 'N/A')}
    Symptom Risk R² Score: {comparison['summary'].get('symptom_risk_r2', 'N/A')}
    Clustering Quality: {comparison['summary'].get('clustering_quality', 'N/A')}
    ========================================
    """
    else:
        error = comparison.get('summary', {}).get('error', 'Unknown error')
        message = f"""
    ========================================
    ML PIPELINE {status}
    ========================================
    Timestamp: {timestamp}
    Error: {error}
    ========================================
    """
    
    print(message)
    
    # In production, send to Discord/Slack/Email
    # Example: requests.post(webhook_url, json={"content": message})
    
    return True


# ============================================================
# MAIN FLOW
# ============================================================
@flow(name="Health ML Pipeline", log_prints=True)
def health_ml_pipeline():
    """
    Complete ML Pipeline for Health Risk Prediction System.
    
    Stages:
    1. Data Ingestion
    2. Data Validation
    3. Feature Engineering
    4. Classification Model Training (Random Forest)
    5. Health Impact Predictor Training (Gradient Boosting for AQI)
    6. Symptom Risk Regression Training (Gradient Boosting)
    7. Clustering Analysis
    8. Model Comparison
    9. Notification
    """
    
    print("=" * 60)
    print("HEALTH ML PIPELINE STARTED")
    print("=" * 60)
    
    try:
        # Stage 1: Load Data
        df = load_data_task()
        
        # Stage 2: Validate Data
        validation = validate_data_task(df)
        
        if not validation["validation_passed"]:
            raise ValueError(f"Data validation failed: {validation.get('error', 'Unknown')}")
        
        # Stage 3: Feature Engineering
        df_engineered, symptom_cols = feature_engineering_task(df)
        
        # Stage 4: Train Classification Model (Random Forest)
        class_results, model_data = train_classification_model_task(df_engineered, symptom_cols)
        
        # Stage 5: Train Health Impact Predictor (Gradient Boosting for AQI)
        health_impact_results = train_health_impact_predictor_task()
        
        # Stage 6: Train Risk Regression Model (Gradient Boosting for symptoms)
        reg_results = train_regression_model_task(df_engineered, symptom_cols)
        
        # Stage 7: Clustering Analysis
        cluster_results = clustering_analysis_task(df_engineered, symptom_cols)
        
        # Stage 8: Compare Models
        comparison = compare_models_task(class_results, reg_results, cluster_results, health_impact_results)
        
        # Stage 8: Notification
        send_notification_task(comparison, success=True)
        
        print("=" * 60)
        print("HEALTH ML PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return comparison
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        send_notification_task({"timestamp": datetime.now().isoformat(), "summary": {"error": str(e)}}, success=False)
        raise


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Run the pipeline
    result = health_ml_pipeline()
    print(f"\nFinal Result: {json.dumps(result, indent=2)}")
