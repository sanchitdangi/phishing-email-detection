"""
diagnose_leakage.py — Dataset diagnostics and leakage investigation.

This script:
1. Loads the processed emails.csv (currently built from the Kaggle dataset).
2. Performs a text check: displays a random sample of 10 legitimate and 10 phishing emails
   to look for artifacts (e.g. leftover headers, email structures, mailing list footers).
3. Evaluates a single-feature baseline: trains a Logistic Regression classifier using
   ONLY the body length (character count) as the input feature to see if this single structural
   signal is trivially predictive.
4. Performs a top-words analysis: fits a quick TF-IDF model and extracts the top 20 terms
   strongly correlated with each class to check if metadata keywords (e.g. "subject", "to",
   "unsubscribe", list names) are acting as shortcuts.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "processed" / "emails.csv"

def run_diagnostics():
    if not DATA_CSV.exists():
        log.error("Processed emails.csv not found. Run download_data.py first.")
        return

    log.info("Loading processed dataset: %s", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    df["text"] = df["text"].fillna("").astype(str)

    # ─── 1. Manual Inspection Sample ──────────────────────────────────────────
    log.info("=" * 70)
    log.info("1. TEXT ARTIFACT INSPECTION")
    log.info("=" * 70)

    for label, name in [(0, "LEGITIMATE (HAM)"), (1, "PHISHING (SPAM)")]:
        sub = df[df["label"] == label].sample(n=5, random_state=42)
        log.info("--- 5 SAMPLE %s EMAILS ---", name)
        for idx, (_, row) in enumerate(sub.iterrows()):
            snippet = row["text"][:350].replace("\n", " ")
            log.info("[%d] (Len=%d) %s...", idx + 1, len(row["text"]), snippet)
            log.info("-" * 40)

    # ─── 2. Body Length Feature Analysis ─────────────────────────────────────
    log.info("=" * 70)
    log.info("2. BODY LENGTH SINGLE-FEATURE BASELINE")
    log.info("=" * 70)

    lengths = df["text"].apply(len).values.reshape(-1, 1)
    labels = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        lengths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Normalise lengths via simple log transform to prevent overflow / help linear solver
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_log, y_train)
    preds = clf.predict(X_test_log)
    probs = clf.predict_proba(X_test_log)[:, 1]

    log.info("Mean length - Legitimate: %.1f chars", df[df["label"] == 0]["text"].apply(len).mean())
    log.info("Mean length - Phishing  : %.1f chars", df[df["label"] == 1]["text"].apply(len).mean())
    log.info("Single-Feature (body_length log) Accuracy: %.4f", accuracy_score(y_test, preds))
    log.info("Single-Feature (body_length log) AUC-ROC : %.4f", roc_auc_score(y_test, probs))
    log.info("Classification Report:\n%s", classification_report(y_test, preds))

    # ─── 3. Top Keywords Correlation (TF-IDF Leakage Check) ─────────────────
    log.info("=" * 70)
    log.info("3. TF-IDF VOCABULARY AND TOP WORDS CORRELATION")
    log.info("=" * 70)

    # Take a subsample of 10,000 emails to speed up vocabulary fitting in diagnostics
    sub_df = df.sample(n=min(10000, len(df)), random_state=42)
    vectoriser = TfidfVectorizer(max_features=2000, stop_words="english")
    X = vectoriser.fit_transform(sub_df["text"])
    y = sub_df["label"].values

    # Train a quick linear model on TF-IDF features to check coefficients
    model = LogisticRegression(C=1.0)
    model.fit(X, y)

    coefs = model.coef_[0]
    feature_names = np.array(vectoriser.get_feature_names_out())

    # Get largest coefficients (pushing toward phishing) and smallest (pushing toward ham)
    top_phish_idx = np.argsort(coefs)[::-1][:20]
    top_ham_idx = np.argsort(coefs)[:20]

    log.info("Top 20 words predicting PHISHING (Spam):")
    for rank, idx in enumerate(top_phish_idx):
        log.info("  Rank %2d: %-15s (coef = %+.4f)", rank + 1, feature_names[idx], coefs[idx])

    log.info("")
    log.info("Top 20 words predicting LEGITIMATE (Ham):")
    for rank, idx in enumerate(top_ham_idx):
        log.info("  Rank %2d: %-15s (coef = %+.4f)", rank + 1, feature_names[idx], coefs[idx])

if __name__ == "__main__":
    run_diagnostics()
