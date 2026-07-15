"""
test_api.py — Integration and endpoint tests for the FastAPI service.
Tests health status, Pydantic inputs validation, multiple classifiers, and rate limiting.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app

@pytest.fixture(scope="module")
def client():
    """Initialise FastAPI test client with lifespan context manager."""
    with TestClient(app) as c:
        yield c

# ─── System / Health Tests ────────────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "loaded_models" in data
    assert "tfidf" in data["loaded_models"]
    assert "lightgbm" in data["loaded_models"]


# ─── Predict Success Cases ───────────────────────────────────────────────────

def test_predict_lightgbm_phishing(client):
    payload = {
        "email_text": "Dear user, please verify your bank account details now at http://192.168.1.1/login immediately!",
        "sender_domain": "chase-banking.com",
        "reply_to_domain": "hacker.com",
        "subject": "Action Required: Account Suspended",
    }
    # Query param defaults to lightgbm
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "prediction_label" in data
    assert "probability" in data
    assert "optimal_threshold" in data
    assert "is_phishing_tuned" in data
    assert "top_features" in data
    
    # Assert type invariants
    assert isinstance(data["prediction"], int)
    assert data["prediction_label"] in ("legitimate", "phishing")
    assert isinstance(data["probability"], float)
    assert isinstance(data["is_phishing_tuned"], bool)
    assert isinstance(data["top_features"], list)


def test_predict_logistic_regression(client):
    payload = {
        "email_text": "Hello Sanchit, thank you for the schedule updates. We will review portland westdesk schedules.",
        "sender_domain": "enron.com",
        "subject": "Schedules Review",
    }
    response = client.post("/predict?model=logistic_regression", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_label"] in ("legitimate", "phishing")
    assert len(data["top_features"]) > 0


def test_predict_naive_bayes(client):
    payload = {
        "email_text": "Normal communication email containing ordinary plain-text words.",
    }
    response = client.post("/predict?model=naive_bayes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_label"] in ("legitimate", "phishing")
    # Naive Bayes doesn't support SHAP explanation output in our pipeline
    assert len(data["top_features"]) == 0


# ─── Validation & Edge Cases ─────────────────────────────────────────────────

def test_predict_invalid_model(client):
    payload = {"email_text": "Generic body."}
    response = client.post("/predict?model=invalid_model_name", json=payload)
    assert response.status_code == 400
    assert "Invalid model name" in response.json()["detail"]


def test_predict_empty_text(client):
    payload = {"email_text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_invalid_domain_format(client):
    payload = {
        "email_text": "Generic body.",
        "sender_domain": "not_a_valid_domain",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
    errors = response.json()["detail"]
    assert any("Domain must be a valid format" in err["msg"] for err in errors)


# ─── Rate Limiting Tests ──────────────────────────────────────────────────────

def test_rate_limiting_predict(client):
    # The limit is set to 10 requests/minute for predict.
    # Trigger 12 calls in a rapid loop to hit HTTP 429
    payload = {"email_text": "Trigger rate limiting text."}
    triggered = False
    for _ in range(15):
        response = client.post("/predict", json=payload)
        if response.status_code == 429:
            triggered = True
            data = response.json()
            assert "error" in data or "detail" in data
            break
    assert triggered, "Rate limiter did not block request at 10/min threshold"
