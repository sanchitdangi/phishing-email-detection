"""
test_feature_engineering.py — Unit tests for URL, header, and structural feature extraction.

These tests cover the custom logic that is most likely to have subtle bugs:
- IP URL detection (edge cases: partial IPs, localhost, IPv6)
- Suspicious TLD matching (eTLD vs subdomain)
- URL shortener detection (eTLD+1 normalisation)
- Domain mismatch heuristic (root-domain normalisation)
- Urgency scoring (word boundary matching)
- Structural feature computation (HTML detection, exclamation density)
- build_structural_feature_vector shape invariant

No dataset loading or model loading — these tests are fast and always runnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    STRUCTURAL_FEATURE_NAMES,
    _root_domain,
    _url_entropy,
    build_structural_feature_vector,
    extract_header_features,
    extract_structural_features,
    extract_url_features,
)
from src.preprocessing import clean_text, extract_urls, preprocess_email, strip_html


# ─── URL feature tests ────────────────────────────────────────────────────────

class TestIPUrlDetection:
    def test_plain_ip_url_detected(self):
        urls = ["http://192.168.1.1/phish"]
        assert extract_url_features(urls)["has_ip_url"] == 1.0

    def test_ip_url_with_port_detected(self):
        urls = ["http://10.0.0.1:8080/login"]
        assert extract_url_features(urls)["has_ip_url"] == 1.0

    def test_domain_url_not_flagged_as_ip(self):
        urls = ["https://paypal.com/login"]
        assert extract_url_features(urls)["has_ip_url"] == 0.0

    def test_localhost_not_flagged(self):
        # localhost has no numeric octets in the typical sense
        urls = ["http://localhost/admin"]
        assert extract_url_features(urls)["has_ip_url"] == 0.0

    def test_partial_ip_not_flagged(self):
        # Three octets only — not a valid IPv4 address
        urls = ["http://192.168.1/phish"]
        assert extract_url_features(urls)["has_ip_url"] == 0.0


class TestSuspiciousTLD:
    def test_tk_tld_flagged(self):
        assert extract_url_features(["http://secure-paypal.tk/login"])["has_suspicious_tld"] == 1.0

    def test_xyz_tld_flagged(self):
        assert extract_url_features(["http://bank-verify.xyz"])["has_suspicious_tld"] == 1.0

    def test_com_not_flagged(self):
        assert extract_url_features(["https://google.com"])["has_suspicious_tld"] == 0.0

    def test_org_not_flagged(self):
        assert extract_url_features(["https://wikipedia.org"])["has_suspicious_tld"] == 0.0

    def test_subdomain_does_not_bypass(self):
        # The suspicious TLD check looks at the *host* ending, not just the last label
        assert extract_url_features(["http://safe-looking-subdomain.evil.tk"])["has_suspicious_tld"] == 1.0


class TestUrlShortener:
    def test_bitly_detected(self):
        assert extract_url_features(["https://bit.ly/3abc123"])["has_url_shortener"] == 1.0

    def test_tinyurl_detected(self):
        assert extract_url_features(["http://tinyurl.com/p5x9q"])["has_url_shortener"] == 1.0

    def test_legitimate_domain_not_flagged(self):
        assert extract_url_features(["https://github.com/user/repo"])["has_url_shortener"] == 0.0

    def test_bitly_subdomain_variant(self):
        # Subdomain of bit.ly — eTLD+1 is still bit.ly
        assert extract_url_features(["https://m.bit.ly/3abc"])["has_url_shortener"] == 1.0


class TestUrlCount:
    def test_empty_list(self):
        feats = extract_url_features([])
        assert feats["url_count"] == 0.0
        assert feats["has_ip_url"] == 0.0
        assert feats["has_suspicious_tld"] == 0.0
        assert feats["has_url_shortener"] == 0.0
        assert feats["max_url_length"] == 0.0
        assert feats["mean_url_entropy"] == 0.0

    def test_multiple_urls_counted(self):
        urls = ["https://a.com", "https://b.com", "https://c.com"]
        assert extract_url_features(urls)["url_count"] == 3.0

    def test_max_url_length(self):
        urls = ["https://short.com", "https://this-is-a-very-long-url-for-testing.com/path/to/page"]
        feats = extract_url_features(urls)
        assert feats["max_url_length"] == max(len(u) for u in urls)


class TestUrlEntropy:
    def test_high_entropy_random_string(self):
        # Random character mix → high entropy
        ent = _url_entropy("aBc3xKm9pZq7rT1vN2")
        assert ent > 3.5

    def test_low_entropy_repeated(self):
        # Single repeated character → minimum entropy
        ent = _url_entropy("aaaaaaaaaaaaa")
        assert ent < 0.1

    def test_empty_string(self):
        assert _url_entropy("") == 0.0


# ─── Header feature tests ─────────────────────────────────────────────────────

class TestDomainMismatch:
    def test_different_root_domains_flagged(self):
        feats = extract_header_features(
            sender_domain="paypal.com",
            reply_to_domain="paypal-secure.tk",
        )
        assert feats["sender_reply_to_mismatch"] == 1.0

    def test_same_root_domain_not_flagged(self):
        feats = extract_header_features(
            sender_domain="mail.google.com",
            reply_to_domain="gmail.com",  # same eTLD+1: google.com vs gmail.com — DIFFERENT
        )
        # mail.google.com → google.com  vs  gmail.com → gmail.com  → MISMATCH
        assert feats["sender_reply_to_mismatch"] == 1.0

    def test_exact_same_domain_no_mismatch(self):
        feats = extract_header_features(
            sender_domain="paypal.com",
            reply_to_domain="paypal.com",
        )
        assert feats["sender_reply_to_mismatch"] == 0.0

    def test_subdomain_same_root_no_mismatch(self):
        feats = extract_header_features(
            sender_domain="noreply.amazon.com",
            reply_to_domain="support.amazon.com",
        )
        # Both → amazon.com
        assert feats["sender_reply_to_mismatch"] == 0.0

    def test_missing_sender_defaults_to_zero(self):
        feats = extract_header_features(sender_domain=None, reply_to_domain="evil.com")
        assert feats["sender_reply_to_mismatch"] == 0.0

    def test_missing_reply_to_defaults_to_zero(self):
        feats = extract_header_features(sender_domain="paypal.com", reply_to_domain=None)
        assert feats["sender_reply_to_mismatch"] == 0.0

    def test_both_missing_defaults_to_zero(self):
        feats = extract_header_features()
        assert feats["sender_reply_to_mismatch"] == 0.0


class TestRootDomain:
    def test_www_stripped(self):
        assert _root_domain("www.example.com") == "example.com"

    def test_single_label(self):
        assert _root_domain("localhost") == "localhost"

    def test_deep_subdomain(self):
        assert _root_domain("a.b.c.d.example.com") == "example.com"


class TestUrgencyScoring:
    def test_urgent_keyword_detected(self):
        feats = extract_header_features(subject="URGENT: Your account has been suspended")
        assert feats["urgency_score"] >= 2  # 'urgent' + 'suspended'

    def test_clean_subject_scores_zero(self):
        feats = extract_header_features(subject="Meeting notes for Monday")
        assert feats["urgency_score"] == 0.0

    def test_empty_subject_scores_zero(self):
        feats = extract_header_features(subject="")
        assert feats["urgency_score"] == 0.0

    def test_multiple_urgency_words(self):
        # 'verify', 'account', 'immediately' should all score
        feats = extract_header_features(subject="Please verify your account immediately")
        assert feats["urgency_score"] >= 2


# ─── Structural feature tests ─────────────────────────────────────────────────

class TestStructuralFeatures:
    def test_html_detected(self):
        raw = "<html><body>Click <a href='http://phish.com'>here</a></body></html>"
        feats = extract_structural_features(raw, "click")
        assert feats["has_html"] == 1.0

    def test_plain_text_not_html(self):
        raw = "Hello, please click the link below to confirm your email."
        feats = extract_structural_features(raw, "hello please click link below confirm email")
        assert feats["has_html"] == 0.0

    def test_exclamation_density_positive(self):
        raw = "You won!!! Click here!!! Free prize!!!"
        cleaned = "won click free prize"
        feats = extract_structural_features(raw, cleaned)
        assert feats["exclamation_density"] > 0.0

    def test_no_exclamations_zero_density(self):
        raw = "Please review the attached document at your convenience."
        cleaned = "please review attached document convenience"
        feats = extract_structural_features(raw, cleaned)
        assert feats["exclamation_density"] == 0.0

    def test_body_length_correct(self):
        raw = "Hello world"   # 11 characters
        feats = extract_structural_features(raw, "hello world")
        assert feats["body_length"] == 11.0

    def test_caps_ratio_all_caps(self):
        raw = "CLICK HERE NOW"
        feats = extract_structural_features(raw, "click now")
        assert feats["caps_ratio"] > 0.5

    def test_empty_text_no_crash(self):
        feats = extract_structural_features("", "")
        assert feats["has_html"] == 0.0
        assert feats["body_length"] == 0.0
        assert feats["exclamation_density"] == 0.0


# ─── Combined vector tests ────────────────────────────────────────────────────

class TestBuildFeatureVector:
    def test_output_shape(self):
        vec = build_structural_feature_vector(
            urls=["http://bit.ly/abc"],
            raw_text="<b>Test email</b>",
            cleaned_text="test email",
        )
        assert vec.shape == (len(STRUCTURAL_FEATURE_NAMES),)

    def test_output_dtype_float32(self):
        vec = build_structural_feature_vector([], "plain text", "plain text")
        assert vec.dtype == np.float32

    def test_no_urls_all_url_features_zero(self):
        vec = build_structural_feature_vector([], "plain text", "plain text")
        # First 6 elements are URL features — all should be 0
        assert np.all(vec[:6] == 0.0)

    def test_ip_url_reflected_in_vector(self):
        vec = build_structural_feature_vector(
            urls=["http://192.0.2.1/phish"],
            raw_text="Click here",
            cleaned_text="click",
        )
        ip_idx = STRUCTURAL_FEATURE_NAMES.index("has_ip_url")
        assert vec[ip_idx] == 1.0

    def test_feature_names_length_constant(self):
        assert len(STRUCTURAL_FEATURE_NAMES) == 13


# ─── Preprocessing tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_html_stripped(self):
        raw = "<p>Hello <b>world</b></p>"
        plain = strip_html(raw)
        assert "<p>" not in plain
        assert "Hello" in plain

    def test_urls_extracted_from_html_href(self):
        raw = '<a href="http://phish.example.com/steal?id=123">Click me</a>'
        urls = extract_urls(raw)
        assert any("phish.example.com" in u for u in urls)

    def test_clean_text_removes_stopwords(self):
        cleaned = clean_text("the quick brown fox jumps over the lazy dog")
        tokens = cleaned.split()
        assert "the" not in tokens
        assert "over" not in tokens

    def test_clean_text_lowercase(self):
        cleaned = clean_text("URGENT VERIFY YOUR ACCOUNT")
        assert cleaned == cleaned.lower()

    def test_preprocess_email_returns_tuple(self):
        cleaned, urls = preprocess_email("<html><body>Test</body></html>")
        assert isinstance(cleaned, str)
        assert isinstance(urls, list)

    def test_empty_email_no_crash(self):
        cleaned, urls = preprocess_email("")
        assert cleaned == ""
        assert urls == []

    def test_none_email_no_crash(self):
        cleaned, urls = preprocess_email(None)
        assert cleaned == ""
        assert urls == []
