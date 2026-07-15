"""
inference.py — Inference manager and model wrapper for FastAPI and Streamlit.
Loads and caches models in memory to ensure fast response times.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_email
from src.feature_engineering import build_structural_feature_vector
from src.explain import explain_email
from src.utils import load_model, load_json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"

# ─── Global model cache ───────────────────────────────────────────────────────

_model_cache: dict[str, Any] = {}


def load_all_models():
    """Load all serialize models into memory during app startup/lifespan."""
    log.info("Lifespan: Pre-loading all classifier models into memory ...")
    
    # 1. TF-IDF Vectorizer
    vectoriser_path = MODELS_DIR / "tfidf_vectoriser.pkl"
    if vectoriser_path.exists():
        _model_cache["tfidf"] = load_model(vectoriser_path)
        log.info("  Loaded TF-IDF Vectorizer.")
    else:
        log.error("  TF-IDF Vectorizer pkl not found at %s", vectoriser_path)

    # 2. LightGBM (Primary)
    lgb_path = MODELS_DIR / "lightgbm_model.pkl"
    if lgb_path.exists():
        _model_cache["lightgbm"] = load_model(lgb_path)
        log.info("  Loaded LightGBM model.")
    else:
        log.error("  LightGBM model pkl not found at %s", lgb_path)

    # 3. Random Forest (Baseline)
    rf_path = MODELS_DIR / "random_forest.pkl"
    if rf_path.exists():
        _model_cache["random_forest"] = load_model(rf_path)
        log.info("  Loaded Random Forest model.")

    # 4. Logistic Regression (Baseline)
    lr_path = MODELS_DIR / "logistic_regression.pkl"
    if lr_path.exists():
        _model_cache["logistic_regression"] = load_model(lr_path)
        log.info("  Loaded Logistic Regression model.")

    # 5. Naive Bayes (Baseline)
    nb_path = MODELS_DIR / "naive_bayes.pkl"
    if nb_path.exists():
        _model_cache["naive_bayes"] = load_model(nb_path)
        log.info("  Loaded Naive Bayes model.")

    # 6. Threshold metadata
    threshold_path = MODELS_DIR / "threshold.json"
    if threshold_path.exists():
        _model_cache["threshold"] = load_json(threshold_path).get("threshold", 0.36)
        log.info("  Loaded tuned threshold: %.2f", _model_cache["threshold"])
    else:
        _model_cache["threshold"] = 0.36
        log.warning("  threshold.json not found. Defaulting to 0.36")


def get_cached_model(name: str) -> Any:
    """Retrieve model from cache, loading it if not cached (lazy load for testing)."""
    if name not in _model_cache:
        load_all_models()
    return _model_cache.get(name)


# ─── Inference Function ───────────────────────────────────────────────────────

def predict_and_explain(
    email_text: str,
    sender_domain: str | None = None,
    reply_to_domain: str | None = None,
    subject: str = "",
    model_name: str = "lightgbm",
) -> dict[str, Any]:
    """
    Runs classification pipeline on a single email and generates SHAP explanations.
    
    Args:
        email_text: Raw body text.
        sender_domain: Domain of the sender email.
        reply_to_domain: Reply-to domain header.
        subject: Email subject.
        model_name: Model to use ('lightgbm', 'random_forest', 'logistic_regression', 'naive_bayes').
    """
    # 1. Retrieve the models
    tfidf = get_cached_model("tfidf")
    model = get_cached_model(model_name)
    optimal_threshold = get_cached_model("threshold")

    if not tfidf or not model:
        raise FileNotFoundError(f"Required models not fully loaded for model_name={model_name}.")

    # 2. Extract features
    cleaned, urls = preprocess_email(email_text)
    
    # Text features (TF-IDF)
    X_tfidf = tfidf.transform([cleaned])
    
    # Naive Bayes is trained on text features only (no structural features)
    if model_name == "naive_bayes":
        X_combined = X_tfidf
    else:
        # Combined features
        X_struct = build_structural_feature_vector(
            urls=urls,
            raw_text=email_text,
            cleaned_text=cleaned,
            sender_domain=sender_domain,
            reply_to_domain=reply_to_domain,
            subject=subject,
        ).reshape(1, -1)
        X_combined = sparse_hstack([X_tfidf, csr_matrix(X_struct)], format="csr")

    # 3. Predict probability and labels
    prob = float(model.predict_proba(X_combined)[:, 1][0])
    pred_default = int(model.predict(X_combined)[0])
    is_phishing_tuned = prob >= optimal_threshold

    # 4. Generate SHAP explanations
    # (Naive Bayes doesn't support SHAP attributions easily, return empty list)
    if model_name == "naive_bayes":
        top_features = []
    else:
        top_features = explain_email(
            email_text=email_text,
            sender_domain=sender_domain,
            reply_to_domain=reply_to_domain,
            subject=subject,
            model_name=model_name,
            top_n=8
        )

    return {
        "prediction": pred_default,
        "prediction_label": "phishing" if pred_default == 1 else "legitimate",
        "probability": round(prob, 4),
        "optimal_threshold": round(optimal_threshold, 2),
        "is_phishing_tuned": bool(is_phishing_tuned),
        "top_features": top_features,
    }
