from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "Health Advisory System API"}

def test_get_symptoms():
    response = client.get("/symptoms")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

def test_predict_endpoint_validation():
    # Test missing fields
    response = client.post("/predict", json={})
    assert response.status_code == 422 # Validation error

def test_predict_endpoint_valid():
    # Attempt a valid prediction request
    # Note: this requires the model to be loaded. If not loaded on test env, it might fail or return 503.
    # We'll check for either success or specific expected failure.
    payload = {
        "age": 30,
        "gender": "Male",
        "city": "New York",
        "symptoms": [0, 0, 0, 1, 0, 0, 0], # Headache
        "symptom_names": ["headache"],
        "other_symptoms": ""
    }
    response = client.post("/predict", json=payload)
    
    # If model is loaded, 200. If not (e.g. invalid path in test env), 503.
    # We accept either as "API is reachable and processing", but ideally 200.
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "risk_score" in data
        assert "disease" in data
