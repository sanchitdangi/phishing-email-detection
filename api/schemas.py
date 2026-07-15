"""
schemas.py — Request and Response validation schemas for the FastAPI backend.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

DOMAIN_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9]"
    r"(?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9]$"
)

class PredictRequest(BaseModel):
    email_text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The raw body text of the email to analyze.",
    )
    sender_domain: Optional[str] = Field(
        None,
        description="The domain of the sender (e.g. bank-security.com).",
    )
    reply_to_domain: Optional[str] = Field(
        None,
        description="The domain in the Reply-To header (e.g. helper.com).",
    )
    subject: str = Field(
        "",
        max_length=1000,
        description="The subject line of the email.",
    )

    @field_validator("sender_domain", "reply_to_domain")
    @classmethod
    def validate_domain(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        value = value.strip().lower()
        if value and not DOMAIN_REGEX.match(value):
            raise ValueError("Domain must be a valid format (e.g., example.com).")
        return value

class ExplanationItem(BaseModel):
    feature: str = Field(..., description="The name of the feature (stemmed token or structural feature).")
    shap_value: float = Field(..., description="The SHAP value indicating feature attribution.")
    direction: str = Field(..., description="Attribution direction: 'phishing' (positive) or 'legitimate' (negative).")

class PredictResponse(BaseModel):
    prediction: int = Field(..., description="The default binary prediction label (0 = ham, 1 = phishing).")
    prediction_label: str = Field(..., description="Legible label: 'legitimate' or 'phishing'.")
    probability: float = Field(..., description="Prediction probability of the phishing class.")
    optimal_threshold: float = Field(..., description="Cost-tuned decision threshold (currently 0.36).")
    is_phishing_tuned: bool = Field(..., description="Binary label under the cost-tuned threshold (probability >= 0.36).")
    top_features: List[ExplanationItem] = Field(..., description="List of the top contributing features sorted by absolute SHAP value.")
