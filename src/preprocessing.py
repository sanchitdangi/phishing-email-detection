"""
preprocessing.py — Email text cleaning and URL extraction pipeline.

Design rationale
----------------
URLs are extracted *before* HTML stripping so that href attributes inside
<a> tags are captured even when the display text is clean prose.  HTML is then
stripped and the resulting plain text is tokenised, lowercased, stopword-
filtered, and stemmed for TF-IDF vectorisation.

Stemming (PorterStemmer) is preferred over lemmatisation here because TF-IDF
only needs vocabulary compression to reduce sparsity — grammatical correctness
is irrelevant for a bag-of-words model.
"""

from __future__ import annotations

import re
import string
from typing import List, Tuple

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ─── NLTK data bootstrap ─────────────────────────────────────────────────────

def _ensure_nltk() -> None:
    """Download required NLTK corpora if not already present (idempotent)."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]
    for data_path, download_id in resources:
        try:
            nltk.data.find(data_path)
        except LookupError:
            nltk.download(download_id, quiet=True)


_ensure_nltk()

# ─── Compiled patterns (built once at import time) ────────────────────────────

# Matches http/https/www URLs; stops at whitespace or common HTML delimiters.
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'\)\(]+",
    re.IGNORECASE,
)

_STEMMER = PorterStemmer()
_STOP_WORDS: set[str] = set(stopwords.words("english"))


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_urls(raw_text: str) -> List[str]:
    """
    Extract all URLs from raw email text (including href values in HTML).

    Called *before* strip_html so that URLs embedded in <a href="..."> tags
    are captured in addition to bare URLs in the plain-text body.
    """
    return _URL_RE.findall(raw_text)


def strip_html(text: str) -> str:
    """
    Remove HTML markup and decode HTML entities.

    Uses BeautifulSoup with the lxml backend for speed and correctness.
    Falls back to html.parser if lxml is not installed.
    """
    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """
    Normalise plain-text email body for TF-IDF vectorisation.

    Pipeline:
      1. Lowercase
      2. Strip punctuation (keep spaces)
      3. Word-tokenise
      4. Drop stopwords and non-alphabetic tokens shorter than 2 chars
      5. PorterStemmer — vocabulary compression, not linguistic accuracy

    Returns a single whitespace-joined string ready for TfidfVectorizer.fit.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok.isalpha() and tok not in _STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)


def preprocess_email(raw_text: str) -> Tuple[str, List[str]]:
    """
    Main preprocessing entry point.

    Args:
        raw_text: The raw email body (may contain HTML, URLs, headers).

    Returns:
        (cleaned_text, urls) where:
          - cleaned_text is ready for TF-IDF vectorisation
          - urls is the list of URLs extracted before stripping
    """
    if not raw_text or not isinstance(raw_text, str):
        return "", []

    urls = extract_urls(raw_text)
    plain = strip_html(raw_text)
    cleaned = clean_text(plain)

    return cleaned, urls
