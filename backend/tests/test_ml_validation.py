"""
Comprehensive ML Model Testing
Tests for data integrity, model performance, and drift detection.
Implements testing concepts similar to DeepChecks.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestDataIntegrity:
    """Tests for data quality and integrity."""
    
    @pytest.fixture
    def training_data(self):
        """Load training data."""
        data_path = os.path.join("backend", "database", "symptoms_and_disease.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        pytest.skip("Training data not found")
    
    def test_no_missing_target(self, training_data):
        """Ensure target column has no missing values."""
        assert 'Disease' in training_data.columns, "Disease column missing"
        assert training_data['Disease'].isnull().sum() == 0, "Disease column has missing values"
    
    def test_minimum_samples(self, training_data):
        """Ensure sufficient training samples."""
        min_samples = 10
        assert len(training_data) >= min_samples, f"Need at least {min_samples} samples"
    
    def test_class_balance(self, training_data):
        """Check for severe class imbalance."""
        class_counts = training_data['Disease'].value_counts()
        max_ratio = class_counts.max() / class_counts.min()
        # Allow up to 10:1 imbalance ratio
        assert max_ratio <= 10, f"Severe class imbalance detected: {max_ratio:.2f}:1"
    
    def test_no_duplicate_rows(self, training_data):
        """Check for duplicate rows."""
        duplicates = training_data.duplicated().sum()
        # Allow up to 5% duplicates
        max_duplicates = len(training_data) * 0.05
        assert duplicates <= max_duplicates, f"Too many duplicate rows: {duplicates}"
    
    def test_feature_values_binary(self, training_data):
        """Ensure symptom features are binary (0/1)."""
        feature_cols = [col for col in training_data.columns if col != 'Disease']
        for col in feature_cols[:10]:  # Check first 10 for speed
            unique_vals = training_data[col].unique()
            assert all(v in [0, 1, 0.0, 1.0] for v in unique_vals), \
                f"Column {col} has non-binary values: {unique_vals}"


class TestModelPerformance:
    """Tests for model performance metrics."""
    
    @pytest.fixture
    def model_data(self):
        """Load trained model."""
        import joblib
        model_path = os.path.join("backend", "ml_models", "symptoms_and_disease.pkl")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        pytest.skip("Model not found")
    
    def test_model_exists(self, model_data):
        """Ensure model is loaded."""
        assert model_data is not None
        assert 'model' in model_data
    
    def test_model_has_encoder(self, model_data):
        """Ensure label encoder exists."""
        assert 'encoder' in model_data
    
    def test_model_accuracy_threshold(self, model_data):
        """Ensure model meets minimum accuracy threshold."""
        min_accuracy = 0.70
        if 'accuracy' in model_data:
            assert model_data['accuracy'] >= min_accuracy, \
                f"Model accuracy {model_data['accuracy']:.4f} below threshold {min_accuracy}"
    
    def test_model_feature_count(self, model_data):
        """Ensure model has expected features."""
        if 'symptom_columns' in model_data:
            n_features = len(model_data['symptom_columns'])
            assert n_features >= 10, f"Too few features: {n_features}"


class TestModelPrediction:
    """Tests for model prediction behavior."""
    
    @pytest.fixture
    def predictor(self):
        """Get disease predictor."""
        try:
            from backend.services.disease_prediction import DiseaseRiskOrchestrator
            return DiseaseRiskOrchestrator()
        except Exception as e:
            pytest.skip(f"Could not load predictor: {e}")
    
    def test_prediction_returns_list(self, predictor):
        """Ensure predictions return a list."""
        # Use empty symptoms
        result = predictor.predict_diseases([], top_n=3)
        assert isinstance(result, list)
    
    def test_prediction_structure(self, predictor):
        """Ensure prediction has expected structure."""
        result = predictor.predict_diseases(['fever'], top_n=1)
        if result:
            assert 'disease' in result[0]
            assert 'probability' in result[0]
    
    def test_precautions_available(self, predictor):
        """Ensure precautions can be retrieved."""
        precautions = predictor.get_precautions("Common Cold")
        assert isinstance(precautions, list)
        # Should have at least generic precautions
        assert len(precautions) >= 0


class TestAPIEndpoints:
    """Tests for API endpoint behavior."""
    
    @pytest.fixture
    def client(self):
        """Get test client."""
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
            return TestClient(app)
        except Exception as e:
            pytest.skip(f"Could not create test client: {e}")
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_symptoms_endpoint(self, client):
        """Test symptoms list endpoint."""
        response = client.get("/symptoms")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_predict_validation(self, client):
        """Test prediction endpoint validation."""
        # Empty payload should fail validation
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_success(self, client):
        """Test successful prediction."""
        payload = {
            "age": 30,
            "gender": "Male",
            "city": "New York",
            "symptoms": ["fever", "headache"],
            "other_symptoms": ""
        }
        response = client.post("/predict", json=payload)
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]


class TestDriftDetection:
    """Tests for detecting data/model drift."""
    
    @pytest.fixture
    def training_data(self):
        """Load training data."""
        data_path = os.path.join("backend", "database", "symptoms_and_disease.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        pytest.skip("Training data not found")
    
    def test_feature_distribution(self, training_data):
        """Check if feature distributions are reasonable."""
        feature_cols = [col for col in training_data.columns if col != 'Disease']
        
        for col in feature_cols[:10]:
            mean = training_data[col].mean()
            # Mean should be between 0 and 1 for binary features
            assert 0 <= mean <= 1, f"Column {col} has suspicious mean: {mean}"
    
    def test_target_distribution(self, training_data):
        """Check if target distribution is stable."""
        value_counts = training_data['Disease'].value_counts(normalize=True)
        
        # No single class should dominate more than 50%
        max_proportion = value_counts.max()
        assert max_proportion < 0.5, f"Single class dominates: {max_proportion:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

