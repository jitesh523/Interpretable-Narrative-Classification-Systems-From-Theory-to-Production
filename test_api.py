from fastapi.testclient import TestClient
from src.api.main import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    # Note: We need the models to be loaded. 
    # TestClient with lifespan context manager handles this automatically in recent FastAPI versions,
    # but we need to ensure the models exist in the expected path.
    
    response = client.post(
        "/predict",
        json={"text": "The spaceship flew to Mars.", "include_explanation": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "genre" in data
    assert "confidence" in data
    assert data["explanation"] == []

def test_predict_with_explanation():
    response = client.post(
        "/predict",
        json={"text": "The spaceship flew to Mars.", "include_explanation": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert "genre" in data
    assert len(data["explanation"]) > 0

if __name__ == "__main__":
    # Run tests manually if executed as script
    with TestClient(app) as client:
        print("Testing /health...")
        test_health()
        print("Testing /predict...")
        test_predict()
        print("Testing /predict with explanation...")
        test_predict_with_explanation()
        print("All tests passed!")
