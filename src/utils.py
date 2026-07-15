"""
utils.py — Model persistence and evaluation logging utilities.

Kept intentionally thin: each function does one thing and is easy to test.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─── Model persistence ────────────────────────────────────────────────────────

def save_model(model: Any, path: Path | str) -> None:
    """Serialise a fitted model to disk with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved → %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_model(path: Path | str) -> Any:
    """
    Load a pickled model from disk.

    Raises FileNotFoundError with a clear message if the artifact is missing,
    so callers can display a graceful fallback rather than letting Python give
    an opaque AttributeError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Run  python src/train.py  to generate it, or see README § Model Artifacts "
            "for download instructions."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── JSON helpers (numpy-safe) ────────────────────────────────────────────────

def _json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serialisable(v) for v in obj]
    return obj


def save_json(data: Any, path: Path | str, *, indent: int = 2) -> None:
    """Write *data* as pretty-printed JSON, handling numpy types transparently."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_serialisable(data), f, indent=indent)
    logger.info("JSON saved → %s", path)


def load_json(path: Path | str) -> Any:
    """Load a JSON file, returning {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Timestamp helpers ────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
