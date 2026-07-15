"""
main.py — FastAPI application orchestrator for phishing email classification.
Includes rate limiting, CORS configuration, custom security headers, and health status.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import PredictRequest, PredictResponse
from api.inference import load_all_models, predict_and_explain, _model_cache
from src.utils import utc_now_iso

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── Lifespan Context Manager ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and clean up at shutdown."""
    load_all_models()
    yield
    _model_cache.clear()
    log.info("Lifespan: Unloaded all models from memory.")


# ─── Rate Limiter Configuration ───────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ─── FastAPI Initialization ───────────────────────────────────────────────────

app = FastAPI(
    title="Phishing Email Detector API",
    description="NLP-driven service classifying emails as legitimate or phishing, with SHAP attribution.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ─── Middleware Setup ─────────────────────────────────────────────────────────

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to portfolio domain in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 2. Custom Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Check health status and list loaded models in the memory cache."""
    loaded_keys = [k for k in _model_cache.keys() if k != "threshold"]
    status_str = "healthy" if len(loaded_keys) >= 2 else "degraded"
    
    return {
        "status": status_str,
        "timestamp": utc_now_iso(),
        "loaded_models": loaded_keys,
        "tuned_threshold": _model_cache.get("threshold", 0.36),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
@limiter.limit("10/minute")
async def predict_email(request: Request, payload: PredictRequest):
    """
    Classify an email raw text body and get local SHAP explainability attributions.
    Rate limit: 10 requests per minute per IP address.
    """
    # 1. Model Selector
    model_name = request.query_params.get("model", "lightgbm").lower()
    valid_models = {"lightgbm", "random_forest", "logistic_regression", "naive_bayes"}
    
    if model_name not in valid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model name. Choose from: {list(valid_models)}",
        )

    # 2. Run prediction
    try:
        t0 = time.time()
        result = predict_and_explain(
            email_text=payload.email_text,
            sender_domain=payload.sender_domain,
            reply_to_domain=payload.reply_to_domain,
            subject=payload.subject,
            model_name=model_name,
        )
        log.info(
            "Prediction request processed in %.2f s using model=%s",
            time.time() - t0,
            model_name,
        )
        return result
    except FileNotFoundError as e_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model files not found or corrupted: {e_model}",
        )
    except Exception as e_proc:
        log.error("Error processing predict request: %s", e_proc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction pipeline: {str(e_proc)}",
        )
