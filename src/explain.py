"""
explain.py — SHAP-based explainability for phishing email predictions.

Design
------
• LightGBM → shap.TreeExplainer  (native support, fast, no background data needed
  with feature_perturbation="tree_path_dependent")
• Random Forest → shap.TreeExplainer  (same approach)
• Logistic Regression → shap.LinearExplainer  (requires background dataset for masker)

For the single-email API use case, only LightGBM explanations are exposed via
/predict, since it is the best-performing model.  The summary plot is generated
over a random 200-sample subset of the test set (computing SHAP for all features
on large datasets is expensive; a representative subsample is standard practice).

Feature name mapping:
    Combined feature index 0 … (vocab_size-1)  →  TF-IDF token names
    Combined feature index vocab_size … vocab_size+12  →  STRUCTURAL_FEATURE_NAMES

Usage:
    python src/explain.py                  # generates SHAP summary plot
    python src/explain.py --sample "..."   # explain a single email
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.sparse import csr_matrix, hstack as sparse_hstack

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import STRUCTURAL_FEATURE_NAMES, build_structural_feature_vector
from src.preprocessing import preprocess_email
from src.utils import load_json, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR  = ROOT / "models"
ASSETS_DIR  = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Explainer loader (cached) ────────────────────────────────────────────────

_explainer_cache: dict = {}


def _get_explainer(model_name: str = "lightgbm") -> shap.Explainer:
    """Load and cache the SHAP explainer for the given model."""
    if model_name in _explainer_cache:
        return _explainer_cache[model_name]

    model_paths = {
        "lightgbm":           MODELS_DIR / "lightgbm_model.pkl",
        "random_forest":      MODELS_DIR / "random_forest.pkl",
        "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
    }
    if model_name not in model_paths:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(model_paths)}")

    model = load_model(model_paths[model_name])

    if model_name in ("lightgbm", "random_forest"):
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            # tree_path_dependent does not require background data and is faster;
            # it may over-estimate feature interactions but is appropriate here.
        )
    elif model_name == "logistic_regression":
        feature_names = load_feature_names()
        dummy_bg = csr_matrix((10, len(feature_names)), dtype=np.float32)
        explainer = shap.LinearExplainer(model, dummy_bg)
    else:
        raise ValueError(f"No SHAP strategy for model: {model_name}")

    _explainer_cache[model_name] = explainer
    return explainer


# ─── Feature name mapping ─────────────────────────────────────────────────────

def load_feature_names() -> list[str]:
    """Load the full ordered feature name list saved during training."""
    path = MODELS_DIR / "feature_names.json"
    names = load_json(path)
    if not names:
        raise FileNotFoundError(
            f"Feature names not found at {path}. Run src/train.py first."
        )
    return names


# ─── Single-email explanation ─────────────────────────────────────────────────

def explain_email(
    email_text: str,
    sender_domain: str | None = None,
    reply_to_domain: str | None = None,
    subject: str = "",
    model_name: str = "lightgbm",
    top_n: int = 10,
) -> list[dict]:
    """
    Explain a single email prediction via SHAP values.

    Returns a list of dicts (sorted by absolute SHAP value descending):
        [{"feature": str, "shap_value": float, "direction": "phishing"|"legitimate"}, ...]

    This is the function called by api/inference.py to populate the
    top_features field in the /predict response.
    """
    vectoriser = load_model(MODELS_DIR / "tfidf_vectoriser.pkl")
    feature_names = load_feature_names()
    explainer = _get_explainer(model_name)

    # Build the same combined feature matrix as in training
    cleaned, urls = preprocess_email(email_text)
    X_tfidf = vectoriser.transform([cleaned])
    X_struct = build_structural_feature_vector(
        urls=urls,
        raw_text=email_text,
        cleaned_text=cleaned,
        sender_domain=sender_domain,
        reply_to_domain=reply_to_domain,
        subject=subject,
    ).reshape(1, -1)
    X_combined = sparse_hstack([X_tfidf, csr_matrix(X_struct)], format="csr")
    X_input = X_combined.toarray() if model_name != "lightgbm" else X_combined

    # Compute SHAP values
    # For LightGBM binary classification: shap_values is shape (n_samples, n_features)
    # (positive = pushes toward phishing, negative = pushes toward legitimate)
    try:
        sv = explainer.shap_values(X_input)
        
        # Handle list vs array vs sparse matrix
        if isinstance(sv, list):
            # List of arrays, e.g. [shap_ham, shap_phishing]
            if len(sv) == 2:
                shap_arr = sv[1][0] if hasattr(sv[1], "toarray") else np.asarray(sv[1])[0]
            else:
                shap_arr = sv[0][0] if hasattr(sv[0], "toarray") else np.asarray(sv[0])[0]
        else:
            # Single array
            sv_arr = sv.toarray() if hasattr(sv, "toarray") else np.asarray(sv)
            if sv_arr.ndim == 3 and sv_arr.shape[2] == 2:
                # Shape (n_samples, n_features, n_classes) -> extract first sample, class 1 (phishing)
                shap_arr = sv_arr[0, :, 1]
            elif sv_arr.ndim == 2:
                # Shape (n_samples, n_features) -> extract first sample
                shap_arr = sv_arr[0]
            else:
                shap_arr = sv_arr.flatten()
    except Exception as exc:
        log.warning("SHAP computation failed: %s — returning empty explanations.", exc)
        return []

    # Map indices → feature names and sort by |SHAP value|
    indices = np.argsort(np.abs(shap_arr))[::-1][:top_n]
    results = []
    for idx in indices:
        sv_val = float(shap_arr[idx])
        if abs(sv_val) < 1e-8:
            continue   # skip negligible values
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        results.append({
            "feature": fname,
            "shap_value": round(sv_val, 6),
            "direction": "phishing" if sv_val > 0 else "legitimate",
        })

    return results


# ─── Summary plot (for README / assets/) ─────────────────────────────────────

def generate_summary_plot(
    n_samples: int = 200,
    model_name: str = "lightgbm",
    top_n_features: int = 20,
) -> Path:
    """
    Generate a SHAP beeswarm summary plot over a random sample of the
    processed dataset.  Saved as PNG to assets/shap_summary.png.

    PNG format (not HTML) prevents the generated-file language contamination
    documented in the companion project.

    Args:
        n_samples: Number of emails to compute SHAP on.  200 is a reasonable
                   sample for a representative plot without excessive runtime.
    """
    import pandas as pd

    data_csv = ROOT / "data" / "processed" / "emails.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found at {data_csv}. Run download_data.py first.")

    log.info("Loading dataset for SHAP summary (sampling %d rows) …", n_samples)
    df = pd.read_csv(data_csv)
    df["subject"] = df.get("subject", pd.Series("", index=df.index)).fillna("").astype(str)
    df["sender_domain"] = df.get("sender_domain", pd.Series("", index=df.index)).fillna("").astype(str)
    df["reply_to_domain"] = df.get("reply_to_domain", pd.Series("", index=df.index)).fillna("").astype(str)
    df = df.sample(n=min(n_samples, len(df)), random_state=42)

    vectoriser   = load_model(MODELS_DIR / "tfidf_vectoriser.pkl")
    feature_names = load_feature_names()
    explainer    = _get_explainer(model_name)

    from tqdm import tqdm
    X_tfidf_rows, X_struct_rows = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Feature extraction"):
        cleaned, urls = preprocess_email(str(row.get("text", "")))
        X_tfidf_rows.append(vectoriser.transform([cleaned]))
        X_struct_rows.append(
            build_structural_feature_vector(
                urls, str(row.get("text", "")), cleaned,
                row.get("sender_domain") or None,
                row.get("reply_to_domain") or None,
                str(row.get("subject", "")),
            ).reshape(1, -1)
        )

    from scipy.sparse import vstack as sparse_vstack
    X_tfidf = sparse_vstack(X_tfidf_rows)
    X_struct = np.vstack(X_struct_rows)
    X_combined = sparse_hstack([X_tfidf, csr_matrix(X_struct)], format="csr")

    log.info("Computing SHAP values for %d samples …", len(df))
    sv = explainer.shap_values(X_combined)
    if isinstance(sv, list) and len(sv) == 2:
        shap_values = sv[1]
    else:
        shap_values = sv

    if hasattr(shap_values, "toarray"):
        shap_values = shap_values.toarray()
    shap_values = np.asarray(shap_values)

    # Identify the top_n_features by mean absolute SHAP value
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n_features]

    feature_names_arr = np.array(feature_names)
    plot_names  = feature_names_arr[top_indices].tolist()
    plot_values = shap_values[:, top_indices]
    plot_data   = X_combined[:, top_indices].toarray()

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        plot_values,
        plot_data,
        feature_names=plot_names,
        show=False,
        plot_type="dot",
        max_display=top_n_features,
    )
    plt.tight_layout()

    out_path = ASSETS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP summary plot saved → %s", out_path)
    return out_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability utilities.")
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Raw email text to explain (single prediction).",
    )
    parser.add_argument(
        "--summary-plot",
        action="store_true",
        help="Generate the SHAP summary plot over a dataset sample.",
    )
    parser.add_argument(
        "--model",
        default="lightgbm",
        choices=["lightgbm", "random_forest", "logistic_regression"],
    )
    args = parser.parse_args()

    if args.sample:
        features = explain_email(args.sample, model_name=args.model)
        log.info("Top features for the provided email:")
        for f in features:
            sign = "→ phishing" if f["direction"] == "phishing" else "→ legitimate"
            log.info("  %+.4f  %-30s  %s", f["shap_value"], f["feature"], sign)

    if args.summary_plot or not args.sample:
        generate_summary_plot(model_name=args.model)
