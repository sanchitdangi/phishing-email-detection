"""
download_data.py — Dataset acquisition for phishing email detection.

Primary source  : SpamAssassin Public Corpus (Apache License 2.0)
                  https://spamassassin.apache.org/old/publiccorpus/
                  Downloadable directly, no credentials required.

Optional source : Kaggle "Phishing Email Dataset" by naserabdullahalam
                  https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
                  Combines Nazario phishing corpus, CEAS 2008, and Enron ham.
                  Requires a Kaggle API key (~/.kaggle/kaggle.json or env vars).

Dataset composition (SpamAssassin path, documented for README provenance):
  • easy_ham  (2,551 emails) — legitimate; Apache-licensed mailing list traffic
  • hard_ham  (250 emails)   — legitimate; borderline messages
  • spam_1    (501 emails)   — spam/phishing mix (2003 corpus)
  • spam_2    (1,397 emails) — spam/phishing mix (2005 corpus)

Honest limitation: The SpamAssassin "spam" class is general spam, not
specifically phishing.  Phishing-specific signals (IP URLs, TLD abuse) are
present but diluted by generic marketing spam.  A dedicated phishing corpus
(Nazario via Kaggle) improves precision on phishing-specific patterns.

Usage:
    python data/download_data.py                  # SpamAssassin (default)
    python data/download_data.py --source kaggle  # Kaggle (requires API key)
    python data/download_data.py --source both    # merge both sources
"""

from __future__ import annotations

import argparse
import email as email_lib
import io
import logging
import sys
import tarfile
from email.policy import default as email_default_policy
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "emails.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── SpamAssassin corpus URLs ─────────────────────────────────────────────────
# Source: https://spamassassin.apache.org/old/publiccorpus/
# License: Apache License 2.0

SA_CORPORA: Dict[str, Dict] = {
    "easy_ham": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "label": 0,       # legitimate
        "source": "spamassassin_easy_ham",
    },
    "hard_ham": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
        "label": 0,
        "source": "spamassassin_hard_ham",
    },
    "spam_1": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "label": 1,       # phishing / spam
        "source": "spamassassin_spam_2003",
    },
    "spam_2": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
        "label": 1,
        "source": "spamassassin_spam_2005",
    },
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _download_bytes(url: str, desc: str) -> bytes:
    """Download a URL with a tqdm progress bar; raises on HTTP error."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)
    return buf.read()


def _parse_email_file(content: bytes) -> Dict[str, str]:
    """
    Parse an RFC 2822 email from raw bytes.

    Returns a dict with keys: subject, sender, reply_to, body.
    All values are strings; missing fields default to "".
    """
    try:
        msg = email_lib.message_from_bytes(content, policy=email_default_policy)
    except Exception:
        return {"subject": "", "sender": "", "reply_to": "", "body": ""}

    subject = str(msg.get("subject") or "")
    sender = str(msg.get("from") or "")
    reply_to = str(msg.get("reply-to") or "")

    body_parts: List[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct in ("text/plain", "text/html"):
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    raw = part.get_payload(decode=True)
                    if raw:
                        body_parts.append(raw.decode("utf-8", errors="replace"))
                break
    else:
        try:
            body_parts.append(msg.get_content())
        except Exception:
            raw = msg.get_payload(decode=True)
            if raw:
                body_parts.append(raw.decode("utf-8", errors="replace"))

    return {
        "subject": subject,
        "sender": sender,
        "reply_to": reply_to,
        "body": "\n".join(body_parts),
    }


def _extract_domain(email_address: str) -> str:
    """
    Pull the domain from a raw email address/header string.

    Handles 'Name <user@domain.com>' and bare 'user@domain.com' forms.
    Returns "" if no '@' found.
    """
    # Strip display name and angle brackets
    match_angle = __import__("re").search(r"<([^>]+)>", email_address)
    if match_angle:
        email_address = match_angle.group(1)
    email_address = email_address.strip()
    if "@" in email_address:
        return email_address.split("@")[-1].lower().strip()
    return ""


# ─── SpamAssassin downloader ─────────────────────────────────────────────────

def download_spamassassin() -> pd.DataFrame:
    """
    Download and parse the SpamAssassin Public Corpus.

    Returns a DataFrame with columns:
        text, label, subject, sender_domain, reply_to_domain, source
    """
    records: List[Dict] = []

    for name, meta in SA_CORPORA.items():
        cache_path = RAW_DIR / f"{name}.tar.bz2"

        if cache_path.exists():
            log.info("Using cached archive: %s", cache_path.name)
            raw = cache_path.read_bytes()
        else:
            log.info("Downloading %s …", meta["url"])
            try:
                raw = _download_bytes(meta["url"], desc=name)
            except Exception as exc:
                log.error("Failed to download %s: %s — skipping.", name, exc)
                continue
            cache_path.write_bytes(raw)

        log.info("Parsing %s …", name)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:bz2") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for member in tqdm(members, desc=f"  {name}", leave=False):
                file_obj = tf.extractfile(member)
                if file_obj is None:
                    continue
                parsed = _parse_email_file(file_obj.read())
                body = parsed["body"].strip()
                if not body:
                    continue
                records.append({
                    "text": body,
                    "label": meta["label"],
                    "subject": parsed["subject"],
                    "sender_domain": _extract_domain(parsed["sender"]),
                    "reply_to_domain": _extract_domain(parsed["reply_to"]),
                    "source": meta["source"],
                })

        log.info("  → %d emails parsed from %s", sum(1 for r in records if r["source"] == meta["source"]), name)

    df = pd.DataFrame(records)
    log.info(
        "SpamAssassin total: %d emails | %d legitimate | %d phishing/spam",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    )
    return df


# ─── Kaggle downloader ────────────────────────────────────────────────────────

def download_kaggle() -> Optional[pd.DataFrame]:
    """
    Download the Kaggle phishing email dataset.

    Dataset: naserabdullahalam/phishing-email-dataset
    Requires either:
      • ~/.kaggle/kaggle.json  with {"username": ..., "key": ...}
      • Environment variables KAGGLE_USERNAME and KAGGLE_KEY

    Returns None if Kaggle credentials are unavailable or download fails,
    with a clear message pointing the user to the manual download path.
    """
    kaggle_dir = RAW_DIR / "kaggle_phishing"
    kaggle_zip = kaggle_dir / "phishing_email.csv"

    if kaggle_zip.exists():
        log.info("Using cached Kaggle dataset: %s", kaggle_zip)
    else:
        # Try kagglehub first (doesn't require credentials)
        try:
            import kagglehub
            import shutil
            log.info("Downloading Kaggle dataset via kagglehub …")
            path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(path) / "phishing_email.csv", kaggle_zip)
            log.info("Successfully cached Kaggle dataset via kagglehub!")
        except Exception as e_hub:
            log.warning("kagglehub download failed: %s. Falling back to kaggle API.", e_hub)
            try:
                import kaggle  # noqa: F401
            except ImportError:
                log.warning(
                    "kaggle package not installed. Run: pip install kaggle\n"
                    "Or download manually and place at %s",
                    kaggle_zip
                )
                return None

            log.info("Downloading Kaggle dataset naserabdullahalam/phishing-email-dataset …")
            try:
                import kaggle as kg
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                kg.api.authenticate()
                kg.api.dataset_download_files(
                    "naserabdullahalam/phishing-email-dataset",
                    path=str(kaggle_dir),
                    unzip=True,
                )
            except Exception as exc:
                log.error(
                    "Kaggle download failed: %s\n"
                    "To download manually:\n"
                    "  1. Go to https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset\n"
                    "  2. Download the CSV and place it at:  %s\n"
                    "  3. Re-run this script.",
                    exc,
                    kaggle_zip,
                )
                return None

    # The Kaggle dataset CSV has columns: Email Text, Email Type
    # Email Type values: "Phishing Email" | "Safe Email"
    try:
        # Try common CSV layouts from this dataset
        df_raw = pd.read_csv(kaggle_zip)
        # Normalise column names
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]

        text_col = next((c for c in df_raw.columns if "text" in c or "body" in c or "email" in c), None)
        label_col = next((c for c in df_raw.columns if "type" in c or "label" in c or "class" in c), None)

        if text_col is None or label_col is None:
            log.error("Unexpected Kaggle CSV columns: %s", df_raw.columns.tolist())
            return None

        df_raw = df_raw[[text_col, label_col]].dropna()
        df_raw.columns = ["text", "raw_label"]

        # Normalise label: phishing=1, ham=0
        def clean_label(val):
            val_str = str(val).strip().lower()
            if val_str in ("1", "1.0", "true", "yes"):
                return 1
            if val_str in ("0", "0.0", "false", "no"):
                return 0
            if "phish" in val_str or "spam" in val_str or "fraud" in val_str:
                return 1
            return 0

        df_raw["label"] = df_raw["raw_label"].apply(clean_label)
        df_raw["subject"] = ""
        df_raw["sender_domain"] = ""
        df_raw["reply_to_domain"] = ""
        df_raw["source"] = "kaggle_phishing_dataset"

        df = df_raw[["text", "label", "subject", "sender_domain", "reply_to_domain", "source"]]
        log.info(
            "Kaggle total: %d emails | %d legitimate | %d phishing",
            len(df),
            (df["label"] == 0).sum(),
            (df["label"] == 1).sum(),
        )
        return df

    except Exception as exc:
        log.error("Failed to parse Kaggle CSV: %s", exc)
        return None


# ─── Main ────────────────────────────────────────────────────────────────────

def build_dataset(source: str = "spamassassin") -> pd.DataFrame:
    """
    Download, combine, and save the email dataset to OUTPUT_CSV.

    Args:
        source: One of "spamassassin" | "kaggle" | "both"

    Returns the combined DataFrame.
    """
    frames: List[pd.DataFrame] = []

    if source in ("spamassassin", "both"):
        sa_df = download_spamassassin()
        if not sa_df.empty:
            frames.append(sa_df)

    if source in ("kaggle", "both"):
        kg_df = download_kaggle()
        if kg_df is not None and not kg_df.empty:
            frames.append(kg_df)

    if not frames:
        log.error(
            "No data was downloaded successfully.\n"
            "Check your network connection, or see README § Dataset for manual steps."
        )
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with empty body text
    combined = combined[combined["text"].str.strip().str.len() > 20].copy()
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Log final composition
    log.info("=" * 60)
    log.info("Final dataset: %d emails", len(combined))
    log.info("  Legitimate : %d  (%.1f%%)", (combined["label"] == 0).sum(),
             100 * (combined["label"] == 0).mean())
    log.info("  Phishing   : %d  (%.1f%%)", (combined["label"] == 1).sum(),
             100 * (combined["label"] == 1).mean())
    log.info("  Sources    : %s", combined["source"].value_counts().to_dict())
    log.info("=" * 60)

    combined.to_csv(OUTPUT_CSV, index=False)
    log.info("Dataset saved → %s", OUTPUT_CSV)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the phishing email dataset."
    )
    parser.add_argument(
        "--source",
        choices=["spamassassin", "kaggle", "both"],
        default="spamassassin",
        help=(
            "spamassassin (default, no credentials needed) | "
            "kaggle (requires ~/.kaggle/kaggle.json) | "
            "both (merge both sources)"
        ),
    )
    args = parser.parse_args()
    build_dataset(args.source)
