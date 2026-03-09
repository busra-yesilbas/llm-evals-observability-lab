"""
Utility functions shared across the LLM Evals Lab package.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np


logger = logging.getLogger(__name__)


# ── String utilities ──────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_overlap_score(a: str, b: str) -> float:
    """
    Compute the Jaccard word-overlap between two strings.

    Returns a float in [0, 1]. Used as a lightweight proxy for
    answer similarity and relevance.
    """
    a_words = set(normalize_text(a).split())
    b_words = set(normalize_text(b).split())
    if not a_words or not b_words:
        return 0.0
    intersection = a_words & b_words
    union = a_words | b_words
    return len(intersection) / len(union)


def token_overlap_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference.

    This is the standard reading-comprehension F1 metric used in SQuAD.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    common = sum(min(pred_counts.get(t, 0), ref_counts.get(t, 0)) for t in ref_counts)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def keyword_coverage(answer: str, key_points: list[str]) -> float:
    """
    Compute what fraction of expected key-point terms appear in the answer.

    Used as a proxy for completeness / recall of key facts.
    """
    if not key_points:
        return 1.0
    answer_norm = normalize_text(answer)
    hits = sum(1 for kp in key_points if normalize_text(kp) in answer_norm)
    return hits / len(key_points)


# ── ID generation ─────────────────────────────────────────────────────────────


def new_run_id() -> str:
    """Generate a UUID4 run identifier."""
    return str(uuid.uuid4())


def make_chunk_id(doc_id: str, order: int) -> str:
    """Construct a deterministic chunk ID from doc_id and chunk order."""
    return f"{doc_id}_chunk_{order:03d}"


def short_hash(text: str, length: int = 8) -> str:
    """Return a short deterministic hash of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


# ── Timing ────────────────────────────────────────────────────────────────────


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """
    Context manager that measures elapsed wall-clock time.

    Usage::

        with timer() as t:
            do_work()
        print(t["ms"])  # elapsed milliseconds
    """
    result: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000.0


# ── Cost estimation ───────────────────────────────────────────────────────────


def estimate_cost(
    prompt_words: int,
    output_words: int,
    input_cost_per_1k: float = 0.0005,
    output_cost_per_1k: float = 0.0015,
    words_to_tokens: float = 1.33,
) -> float:
    """
    Estimate USD cost given word counts and per-token rates.

    Note: In local/fallback mode this is always $0.0. Cost estimates
    are only meaningful when using a paid LLM API backend.
    """
    prompt_tokens = prompt_words * words_to_tokens
    output_tokens = output_words * words_to_tokens
    cost = (prompt_tokens / 1000) * input_cost_per_1k
    cost += (output_tokens / 1000) * output_cost_per_1k
    return round(cost, 6)


# ── JSON helpers ──────────────────────────────────────────────────────────────


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    """Save a JSON-serializable object to a file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for non-standard types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


# ── DataFrame helpers ─────────────────────────────────────────────────────────


def flatten_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a list of records without nested structures (best-effort)."""
    flat = []
    for rec in records:
        flat.append({k: v for k, v in rec.items() if not isinstance(v, (dict, list))})
    return flat


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logging for the lab."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )
