"""
Vector index for chunk retrieval.

Stores chunk embeddings as a numpy matrix and provides nearest-neighbour
search via dot-product (cosine) similarity. All vectors are assumed to be
L2-normalised (handled by the embedder).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from llm_evals_lab.retrieval.embedder import BaseEmbedder
from llm_evals_lab.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class ChunkIndex:
    """
    In-memory nearest-neighbour index over document chunks.

    Uses a dense numpy matrix for storage and dot-product similarity for
    search. Suitable for corpora up to ~50k chunks without performance issues.

    Parameters
    ----------
    embedder : BaseEmbedder
        Fitted embedding backend used to produce query vectors.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self._chunks: list[DocumentChunk] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, D)
        self._built = False

    # ── Index construction ────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk]) -> None:
        """
        Fit the embedder (if needed) and build the index.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            All chunks to index.
        """
        self._chunks = chunks
        texts = [c.chunk_text for c in chunks]

        # Fit the embedder if it hasn't been fitted yet (TF-IDF)
        if hasattr(self.embedder, "_fitted") and not self.embedder._fitted:
            self.embedder.fit(texts)

        logger.info("Building index for %d chunks...", len(chunks))
        self._vectors = self.embedder.embed(texts)  # (N, D)
        self._built = True
        logger.info("Index built. Vector shape: %s", self._vectors.shape)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve top-k most similar chunks to the query.

        Returns
        -------
        list of (DocumentChunk, score) tuples, sorted descending by score.
        Scores are in [0, 1] (cosine similarity of L2-normalised vectors).
        """
        if not self._built or self._vectors is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        query_vec = self.embedder.embed_query(query)  # (D,)
        scores = self._vectors @ query_vec  # (N,) — dot product = cosine similarity

        # Clip to [0, 1]: TF-IDF scores are always >= 0; ST scores can be negative
        scores = np.clip(scores, 0.0, 1.0)

        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold:
                results.append((self._chunks[idx], score))

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize the index (chunks + vectors + embedder) to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._vectors,
            "embedder_name": self.embedder.name,
            "embedder": self.embedder,  # serialise the fitted embedder
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Index saved to %s (%d chunks)", path, len(self._chunks))

    @classmethod
    def load(cls, path: Path, embedder: BaseEmbedder) -> "ChunkIndex":
        """Load a serialized index from disk.

        The embedder stored in the payload takes precedence over the
        ``embedder`` argument so that fitted state (e.g. TF-IDF vocabulary)
        is correctly restored.
        """
        with path.open("rb") as fh:
            payload = pickle.load(fh)

        # Restore the fitted embedder if present in the payload
        restored_embedder = payload.get("embedder", embedder)
        obj = cls(restored_embedder)
        obj._chunks = [DocumentChunk(**r) for r in payload["chunks"]]
        obj._vectors = payload["vectors"]
        obj._built = True
        logger.info("Index loaded from %s (%d chunks)", path, len(obj._chunks))
        return obj

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_built(self) -> bool:
        return self._built
