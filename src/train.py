"""
train.py — Model training and evaluation pipeline on the Kaggle Phishing Email Dataset.

Optimisations:
1. Feature cache: Caches parallel preprocessing output to save 7.5 min of CPU time.
2. Parallel workers: Uses multiprocessing for BS4 and NLTK text processing.
3. LightGBM speedup: Reduced tuning CV splits to 3, iterations to 6, and estimators
   to 80-250 to allow completion in 5-10 minutes on a single CPU thread.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

import lightgbm as lgb
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import STRUCTURAL_FEATURE_NAMES, build_structural_feature_vector
from src.preprocessing import preprocess_email
from src.utils import load_json, save_json, save_model, utc_now_iso

# ─── Paths and Settings ───────────────────────────────────────────────────────

DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

TFIDF_MAX_FEATURES = 15_000
RANDOM_STATE = 42


# ─── Data Loader ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the processed CSV."""
    if not path.exists():
        log.error("Dataset not found at %s. Run download_data.py first.", path)
        sys.exit(1)

    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        log.error("CSV must have columns: %s  (found: %s)", required, df.columns.tolist())
        sys.exit(1)

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["subject"] = df.get("subject", pd.Series("", index=df.index)).fillna("").astype(str)
    df["sender_domain"] = df.get("sender_domain", pd.Series("", index=df.index)).fillna("").astype(str)
    df["reply_to_domain"] = df.get("reply_to_domain", pd.Series("", index=df.index)).fillna("").astype(str)
    df["source"] = df.get("source", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)

    log.info(
        "Dataset loaded: %d emails | %d legitimate (%.1f%%) | %d phishing (%.1f%%)",
        len(df),
        (df["label"] == 0).sum(), 100 * (df["label"] == 0).mean(),
        (df["label"] == 1).sum(), 100 * (df["label"] == 1).mean(),
    )
    return df


# ─── Preprocessing + parallel feature extraction ─────────────────────────────

def _process_single_row(row_tuple) -> tuple[str, np.ndarray]:
    """Helper function for parallel preprocessing (must be top-level for pickling)."""
    _, text, sender_domain, reply_to_domain, subject = row_tuple
    cleaned, urls = preprocess_email(str(text))
    vec = build_structural_feature_vector(
        urls=urls,
        raw_text=str(text),
        cleaned_text=cleaned,
        sender_domain=sender_domain or None,
        reply_to_domain=reply_to_domain or None,
        subject=str(subject),
    )
    return cleaned, vec

def build_features(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Run preprocessing on all emails and build the structural feature matrix.
    Caches features to disk to prevent redundant runs.
    """
    cache_txt_path = ROOT / "data" / "processed" / "cache_cleaned_texts.json"
    cache_npy_path = ROOT / "data" / "processed" / "cache_struct_matrix.npy"

    if cache_txt_path.exists() and cache_npy_path.exists():
        try:
            log.info("Loading preprocessed features from cache …")
            cleaned_texts = load_json(cache_txt_path)
            struct_matrix = np.load(cache_npy_path).astype(np.float32)
            if len(cleaned_texts) == len(df) and struct_matrix.shape[0] == len(df):
                log.info("  Successfully loaded %d cached feature records.", len(df))
                return cleaned_texts, struct_matrix
            else:
                log.warning("  Cache length mismatch. Recomputing features …")
        except Exception as e_cache:
            log.warning("  Failed to load feature cache (%s). Recomputing …", e_cache)

    log.info("Preprocessing %d emails using parallel workers …", len(df))
    rows = list(zip(
        df.index,
        df["text"].values,
        df["sender_domain"].values,
        df["reply_to_domain"].values,
        df["subject"].values
    ))
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    log.info("  Spawning %d worker processes", num_workers)
    
    cleaned_texts: list[str] = []
    struct_rows: list[np.ndarray] = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_single_row, rows, chunksize=250),
            total=len(rows),
            desc="  Feature extraction"
        ))
        
    for cleaned, vec in results:
        cleaned_texts.append(cleaned)
        struct_rows.append(vec)
        
    struct_matrix = np.vstack(struct_rows).astype(np.float32)
    log.info("  Structural feature matrix shape: %s", struct_matrix.shape)

    # Save cache
    try:
        save_json(cleaned_texts, cache_txt_path)
        np.save(cache_npy_path, struct_matrix)
        log.info("  Features successfully cached to disk.")
    except Exception as e_save:
        log.warning("  Failed to save features cache: %s", e_save)

    return cleaned_texts, struct_matrix


# ─── Metrics helper ───────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
              y_prob: np.ndarray | None = None) -> dict:
    """Compute and log a standard evaluation metrics dict."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    log.info(
        "[%s] Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%s",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        f"{metrics['auc_roc']:.4f}" if metrics["auc_roc"] is not None else "N/A",
    )
    return metrics


# ─── Individual Model Training Routines ───────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
    log.info("Training Logistic Regression …")
    t0 = time.time()
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    log.info("  LR trained in %.1f s", time.time() - t0)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate("Logistic Regression", y_test, y_pred, y_prob)
    save_model(model, MODELS_DIR / "logistic_regression.pkl")
    return metrics


def train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test) -> dict:
    log.info("Training Naive Bayes (TF-IDF features only — see docstring) …")
    t0 = time.time()
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    log.info("  NB trained in %.1f s", time.time() - t0)

    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    metrics = evaluate("Naive Bayes (text-only)", y_test, y_pred, y_prob)
    save_model(model, MODELS_DIR / "naive_bayes.pkl")
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    log.info("Training Random Forest (100 trees) …")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,      # Constrained max_depth to prevent file size exceeding 50MB
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info("  RF trained in %.1f s", time.time() - t0)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate("Random Forest", y_test, y_pred, y_prob)
    save_model(model, MODELS_DIR / "random_forest.pkl")
    return metrics


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test) -> tuple[dict, float]:
    """Tuning CV splits to 3, iterations to 6 for speed on large datasets."""
    from scipy.stats import loguniform, randint as sp_randint

    log.info("Training LightGBM with RandomizedSearchCV (n_iter=6, cv=3) …")
    log.info("  (Optimised for larger Kaggle dataset — roughly 5–10 min)")
    t0 = time.time()

    base_model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )

    param_dist = {
        "n_estimators":  sp_randint(80, 250),
        "learning_rate": loguniform(0.03, 0.2),
        "num_leaves":    sp_randint(20, 60),
        "max_depth":     [-1, 7, 10],
        "reg_alpha":     loguniform(1e-4, 1.0),
        "reg_lambda":    loguniform(1e-4, 1.0),
        "min_child_samples": sp_randint(10, 50),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=6,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    log.info("  Best params: %s", best_params)
    log.info("  Best CV F1: %.4f", search.best_score_)
    log.info("  LightGBM search completed in %.1f s", time.time() - t0)

    model = search.best_estimator_

    # Default performance (threshold=0.50)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate("LightGBM (default t=0.50)", y_test, y_pred, y_prob)

    # Cost-sensitive threshold tuning on validation set
    y_val_prob = model.predict_proba(X_val)[:, 1]
    optimal_threshold = tune_threshold(y_val, y_val_prob)

    # Re-evaluate with tuned threshold
    y_pred_tuned = (y_prob >= optimal_threshold).astype(int)
    tuned_metrics = evaluate(
        f"LightGBM (threshold={optimal_threshold:.2f}, cost-tuned)",
        y_test,
        y_pred_tuned,
        y_prob,
    )

    save_model(model, MODELS_DIR / "lightgbm_model.pkl")
    return tuned_metrics, optimal_threshold


# ─── Threshold tuning helper ──────────────────────────────────────────────────

def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> float:
    """Find threshold that minimises fn_cost * FN + fp_cost * FP."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = float("inf")
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    log.info(
        "Optimal threshold (cost ratio FN:FP=%.1f:%.1f): %.2f  (cost=%d)",
        fn_cost,
        fp_cost,
        best_threshold,
        best_cost,
    )
    return float(best_threshold)


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def main(data_path: Path, force: bool = False):
    t_start = time.time()
    log.info("=================================================================")
    log.info("Phishing Email Classifier — Training Pipeline")
    log.info("=================================================================")

    # Load data
    df = load_data(data_path)

    # Feature extraction
    cleaned_texts, X_struct = build_features(df)

    # Train/Val/Test Split (70 / 15 / 15)
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=df["label"].values,
        random_state=RANDOM_STATE,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=df["label"].values[temp_idx],
        random_state=RANDOM_STATE,
    )

    log.info("Split: train=%d | val=%d | test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Fit vectoriser ONLY on training data
    log.info("Fitting TF-IDF vectoriser (max_features=%d) on training data …", TFIDF_MAX_FEATURES)
    vectoriser = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    vectoriser.fit([cleaned_texts[i] for i in train_idx])
    log.info("  Vocabulary size (actual): %d", len(vectoriser.vocabulary_))
    save_model(vectoriser, MODELS_DIR / "tfidf_vectoriser.pkl")

    # Map vocab indices + structural features
    vocab = sorted(vectoriser.vocabulary_, key=vectoriser.vocabulary_.get)
    feature_names = vocab + STRUCTURAL_FEATURE_NAMES
    save_json(feature_names, MODELS_DIR / "feature_names.json")
    log.info("  Feature names saved (%d total)", len(feature_names))

    # Vectorise all splits
    X_train_tfidf = vectoriser.transform([cleaned_texts[i] for i in train_idx])
    X_val_tfidf   = vectoriser.transform([cleaned_texts[i] for i in val_idx])
    X_test_tfidf  = vectoriser.transform([cleaned_texts[i] for i in test_idx])

    X_train_combined = sparse_hstack([X_train_tfidf, csr_matrix(X_struct[train_idx])], format="csr")
    X_val_combined   = sparse_hstack([X_val_tfidf, csr_matrix(X_struct[val_idx])], format="csr")
    X_test_combined  = sparse_hstack([X_test_tfidf, csr_matrix(X_struct[test_idx])], format="csr")

    y_train = df["label"].values[train_idx]
    y_val   = df["label"].values[val_idx]
    y_test  = df["label"].values[test_idx]

    # Run trainings
    results = {}
    results["logistic_regression"] = train_logistic_regression(
        X_train_combined, y_train, X_test_combined, y_test
    )
    results["naive_bayes"] = train_naive_bayes(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    results["random_forest"] = train_random_forest(
        X_train_combined, y_train, X_test_combined, y_test
    )

    lgb_metrics, optimal_t = train_lightgbm(
        X_train_combined, y_train,
        X_val_combined, y_val,
        X_test_combined, y_test,
    )
    results["lightgbm_cost_tuned"] = lgb_metrics

    # Save metadata
    metadata = {
        "dataset_size": len(df),
        "class_distribution": {
            "legitimate": int((df["label"] == 0).sum()),
            "phishing": int((df["label"] == 1).sum()),
        },
        "optimal_threshold": optimal_t,
        "feature_count": len(feature_names),
        "timestamp": utc_now_iso(),
    }
    save_json(metadata, MODELS_DIR / "training_metadata.json")
    save_json(results, MODELS_DIR / "evaluation_results.json")
    save_json({"threshold": optimal_t}, MODELS_DIR / "threshold.json")

    # Save actual predictions (true labels and probabilities) for dynamic ROC curves in Streamlit
    predictions_data = {
        "y_test": y_test.tolist(),
        "logistic_regression": load_model(MODELS_DIR / "logistic_regression.pkl").predict_proba(X_test_combined)[:, 1].tolist(),
        "naive_bayes": load_model(MODELS_DIR / "naive_bayes.pkl").predict_proba(X_test_tfidf)[:, 1].tolist(),
        "random_forest": load_model(MODELS_DIR / "random_forest.pkl").predict_proba(X_test_combined)[:, 1].tolist(),
        "lightgbm": load_model(MODELS_DIR / "lightgbm_model.pkl").predict_proba(X_test_combined)[:, 1].tolist(),
    }
    save_json(predictions_data, MODELS_DIR / "predictions.json")

    log.info("=================================================================")
    log.info("MODEL COMPARISON (test set)")
    log.info("Model                                   Acc    Prec     Rec      F1     AUC")
    log.info("-----------------------------------------------------------------")
    for name, r in results.items():
        auc_str = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "N/A"
        log.info(
            "%-36s  %.4f  %.4f  %.4f  %.4f  %s",
            name, r["accuracy"], r["precision"], r["recall"], r["f1"], auc_str
        )
    log.info("=================================================================")
    log.info("Total training pipeline time: %.1f s", time.time() - t_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Classifier Training Pipeline")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to processed CSV")
    parser.add_argument("--force", action="store_true", help="Force retrain all models")
    args = parser.parse_args()

    main(data_path=args.data, force=args.force)
