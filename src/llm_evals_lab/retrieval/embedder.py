"""
Embedding abstraction for the LLM Evals Lab.

Provides two backends:
  1. TF-IDF (default) — no external dependencies, always available.
  2. SentenceTransformers — richer semantic embeddings, optional.

The factory function ``get_embedder`` checks availability and falls back
gracefully so the full pipeline always runs.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all embedding backends."""

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Fit the embedder on a corpus of texts."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Return an (N, D) float32 array of embeddings.
        Rows are L2-normalised so dot-product == cosine similarity.
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning shape (D,)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend."""


class TFIDFEmbedder(BaseEmbedder):
    """
    TF-IDF based embedding using scikit-learn.

    Vectors are L2-normalised so cosine similarity reduces to dot product.
    Supports bigrams to capture short phrases.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english",
            strip_accents="unicode",
        )
        self._fitted = False

    @property
    def name(self) -> str:
        return "tfidf"

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vocabulary on the corpus."""
        self._vectorizer.fit(texts)
        self._fitted = True
        logger.info("TFIDFEmbedder fitted on %d documents", len(texts))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised TF-IDF vectors, shape (N, vocab_size)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return normalize(matrix, norm="l2")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        vec = self.embed([query])
        return vec[0]

    def save(self, path: Path) -> None:
        """Persist the fitted vectorizer to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self._vectorizer, fh)
        logger.debug("TFIDFEmbedder saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbedder":
        """Load a previously fitted vectorizer."""
        with path.open("rb") as fh:
            vectorizer = pickle.load(fh)
        obj = cls.__new__(cls)
        obj._vectorizer = vectorizer
        obj._fitted = True
        logger.debug("TFIDFEmbedder loaded from %s", path)
        return obj


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence-transformers semantic embedding backend.

    Falls back gracefully: instantiation raises ``ImportError`` if the
    library is not installed, allowing callers to catch and fall back to TF-IDF.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers  "
                "or use the TF-IDF backend."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("SentenceTransformerEmbedder loaded: %s", model_name)

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    def fit(self, texts: list[str]) -> None:
        """No-op: sentence-transformers are pretrained."""
        logger.debug("SentenceTransformerEmbedder.fit() is a no-op (pretrained model)")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings."""
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vecs = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return vecs[0].astype(np.float32)


def get_embedder(
    backend: str = "tfidf",
    model_name: str = "all-MiniLM-L6-v2",
    tfidf_max_features: int = 10_000,
) -> BaseEmbedder:
    """
    Factory: return the requested embedder, falling back to TF-IDF if
    sentence-transformers is unavailable.

    Parameters
    ----------
    backend : str
        "tfidf" or "sentence-transformers"
    model_name : str
        Model name for sentence-transformers backend.
    tfidf_max_features : int
        Vocabulary size for TF-IDF backend.
    """
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not available — falling back to TF-IDF."
            )
    return TFIDFEmbedder(max_features=tfidf_max_features)
