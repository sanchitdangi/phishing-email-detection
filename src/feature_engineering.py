"""
feature_engineering.py — URL, header, and structural feature extraction.

This module is the main differentiator from text-only phishing classifiers.
Each feature is grounded in published phishing-detection or security-economics
literature — citations are inline below.

Feature categories
------------------
1. URL features (6)
   Phishing emails characteristically abuse link structure in ways that
   legitimate senders do not (Sahoo et al., 2019; APWG eCrime reports).

2. Header / metadata features (2)
   sender_domain / reply_to_domain mismatch and urgency-word scoring.
   These are optional inputs: when the caller only has the email body,
   both default to 0.  This is an explicit limitation (documented in README):
   metadata features improve precision but require header access.

3. Structural features (5)
   Coarse body-level signals: presence of HTML, body length, word count,
   exclamation density, and ALLCAPS ratio.

Canonical feature order
-----------------------
STRUCTURAL_FEATURE_NAMES defines the fixed ordering of the 13-element dense
vector returned by build_structural_feature_vector().  Any change to this
list requires retraining all saved models — the constant acts as a contract
between feature engineering and inference.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np


# ─── Reference data ──────────────────────────────────────────────────────────

# TLDs heavily abused for phishing — sourced from:
#   • APWG Phishing Activity Trends Report Q4 2023 (top abused TLDs table)
#   • Spamhaus Domain Block List (DBL) documentation on "newly registered" TLDs
# Freenom discontinued free registrations in 2023 but legacy .tk/.ml/.ga/.cf/.gq
# domains remain active in the wild and still appear in phishing datasets.
SUSPICIOUS_TLDS: frozenset[str] = frozenset({
    ".tk", ".ml", ".ga", ".cf", ".gq",   # Freenom legacy free TLDs
    ".xyz", ".top", ".click", ".link",
    ".work", ".online", ".site", ".tech",
    ".club", ".buzz", ".icu", ".live",
    ".vip", ".bid", ".loan", ".win",
})

# Known URL-shortener domains (source: various public threat-intel lists and
# APWG operational reports citing shorteners as obfuscation vectors).
URL_SHORTENERS: frozenset[str] = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "goo.gl",
    "is.gd", "cli.gs", "dlvr.it", "ift.tt", "tiny.cc",
    "lnkd.in", "db.tt", "short.to", "rb.gy", "cutt.ly",
    "shorturl.at", "tr.im", "v.gd",
})

# Urgency / social-engineering vocabulary.
# Source: NIST SP 800-177 Rev. 1 "Trustworthy Email" § 4.6 lists urgency
# and authority as primary social-engineering vectors; the specific word list
# below is derived from that framework combined with APWG annual report
# keyword analyses.
URGENCY_WORDS: frozenset[str] = frozenset({
    "urgent", "immediately", "verify", "expire", "expires", "expired",
    "suspended", "limited", "winner", "congratulations", "free", "prize",
    "confirm", "click", "login", "update", "account", "password",
    "security", "unauthorized", "unusual", "suspicious", "blocked",
    "locked", "restricted", "action", "required", "attention", "alert",
    "warning", "critical", "important", "overdue", "verify", "validate",
})

# Compiled patterns
_IP_IN_HOST_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}(:\d+)?$")
_HTML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")


# ─── URL features ─────────────────────────────────────────────────────────────

def _url_entropy(url: str) -> float:
    """
    Shannon entropy of characters in a URL.

    High entropy in the URL path is a signal of obfuscation — phishers use
    random-looking token strings to evade signature-based detection.
    Legitimate URLs tend to use human-readable path components.
    """
    if not url:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in url:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(url)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def extract_url_features(urls: List[str]) -> Dict[str, float]:
    """
    Derive phishing-relevant numeric features from a list of URL strings.

    Returns a dict with keys matching the first six entries of
    STRUCTURAL_FEATURE_NAMES.  Safe to call with an empty list.
    """
    if not urls:
        return {
            "url_count": 0.0,
            "has_ip_url": 0.0,
            "has_suspicious_tld": 0.0,
            "has_url_shortener": 0.0,
            "max_url_length": 0.0,
            "mean_url_entropy": 0.0,
        }

    has_ip = 0.0
    has_susp_tld = 0.0
    has_shortener = 0.0
    lengths: List[int] = []
    entropies: List[float] = []

    for url in urls:
        # Normalise for urlparse
        normalised = url if "://" in url else "http://" + url
        try:
            parsed = urlparse(normalised)
            host = parsed.netloc.lower().split(":")[0]  # strip port
        except Exception:
            host = ""

        # IP-based host (e.g. http://192.0.2.1/phish)
        if _IP_IN_HOST_RE.match(host):
            has_ip = 1.0

        # Suspicious TLD — match against the host's effective TLD
        for tld in SUSPICIOUS_TLDS:
            if host.endswith(tld):
                has_susp_tld = 1.0
                break

        # URL shortener — match the eTLD+1 (e.g. "bit.ly")
        parts = host.split(".")
        etld_plus1 = ".".join(parts[-2:]) if len(parts) >= 2 else host
        if etld_plus1 in URL_SHORTENERS:
            has_shortener = 1.0

        lengths.append(len(url))
        entropies.append(_url_entropy(url))

    return {
        "url_count": float(len(urls)),
        "has_ip_url": has_ip,
        "has_suspicious_tld": has_susp_tld,
        "has_url_shortener": has_shortener,
        "max_url_length": float(max(lengths)),
        "mean_url_entropy": float(np.mean(entropies)),
    }


# ─── Header / metadata features ───────────────────────────────────────────────

def _root_domain(domain: str) -> str:
    """
    Reduce a domain string to its eTLD+1 (e.g. 'mail.paypal.com' → 'paypal.com').

    Simple implementation: take the last two dot-separated labels.
    Does not handle multi-part TLDs (e.g. .co.uk) — sufficient for the
    mismatch heuristic where false negatives are tolerable.
    """
    parts = domain.strip().lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain.lower().strip()


def extract_header_features(
    sender_domain: Optional[str] = None,
    reply_to_domain: Optional[str] = None,
    subject: str = "",
) -> Dict[str, float]:
    """
    Features derived from email header fields.

    sender_domain / reply_to_domain are Optional because the API allows
    body-only predictions.  When absent, both header features default to 0.
    This is an honest limitation: header-based features improve precision
    but cannot be computed without header access.

    Limitation note (documented in README): the root-domain mismatch check
    uses a naive eTLD+1 heuristic that fails on multi-part TLDs (.co.uk etc.)
    and on legitimate senders that intentionally use different reply-to domains
    for tracking (e.g. some marketing platforms).  False positive rate for this
    single feature is therefore non-trivial; it is most useful in combination
    with other signals.
    """
    mismatch = 0.0
    if sender_domain and reply_to_domain:
        if _root_domain(sender_domain) != _root_domain(reply_to_domain):
            mismatch = 1.0

    # Urgency score: count distinct urgency words in the subject line.
    subject_tokens = set(re.sub(r"[^a-z\s]", "", subject.lower()).split())
    urgency_score = float(len(subject_tokens & URGENCY_WORDS))

    return {
        "sender_reply_to_mismatch": mismatch,
        "urgency_score": urgency_score,
    }


# ─── Structural / body features ───────────────────────────────────────────────

def extract_structural_features(
    raw_text: str,
    cleaned_text: str,
) -> Dict[str, float]:
    """
    Coarse body-level features that complement the TF-IDF representation.

    Args:
        raw_text:     Original email body before any preprocessing.
        cleaned_text: Preprocessed, stemmed text (output of preprocessing.clean_text).
    """
    has_html = 1.0 if _HTML_TAG_RE.search(raw_text) else 0.0
    body_length = float(len(raw_text))

    words = cleaned_text.split() if cleaned_text else []
    word_count = float(len(words))

    excl_count = raw_text.count("!")
    exclamation_density = (excl_count / word_count * 100.0) if word_count > 0 else 0.0

    alpha_chars = [c for c in raw_text if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )

    return {
        "has_html": has_html,
        "body_length": body_length,
        "word_count": word_count,
        "exclamation_density": exclamation_density,
        "caps_ratio": caps_ratio,
    }


# ─── Combined vector ──────────────────────────────────────────────────────────

# Canonical feature name ordering — this is the contract between feature
# engineering and inference.  Changing the order or adding/removing names
# requires retraining all saved models.
STRUCTURAL_FEATURE_NAMES: List[str] = [
    # URL features (6)
    "url_count",
    "has_ip_url",
    "has_suspicious_tld",
    "has_url_shortener",
    "max_url_length",
    "mean_url_entropy",
    # Header features (2)
    "sender_reply_to_mismatch",
    "urgency_score",
    # Structural features (5)
    "has_html",
    "body_length",
    "word_count",
    "exclamation_density",
    "caps_ratio",
]

assert len(STRUCTURAL_FEATURE_NAMES) == 13, "Feature name list length must be 13"


def build_structural_feature_vector(
    urls: List[str],
    raw_text: str,
    cleaned_text: str,
    sender_domain: Optional[str] = None,
    reply_to_domain: Optional[str] = None,
    subject: str = "",
) -> np.ndarray:
    """
    Build a 13-element float32 feature vector for a single email.

    The returned vector follows the order of STRUCTURAL_FEATURE_NAMES.
    This is the dense component that is hstacked with the sparse TF-IDF
    matrix in train.py to form the combined feature matrix.

    scipy.sparse.hstack is used (not numpy.hstack) to avoid materialising a
    dense TF-IDF matrix in memory — important when vocab_size ~ 15,000.
    """
    url_feats = extract_url_features(urls)
    header_feats = extract_header_features(sender_domain, reply_to_domain, subject)
    struct_feats = extract_structural_features(raw_text, cleaned_text)

    combined = {**url_feats, **header_feats, **struct_feats}
    return np.array(
        [combined[name] for name in STRUCTURAL_FEATURE_NAMES],
        dtype=np.float32,
    )
