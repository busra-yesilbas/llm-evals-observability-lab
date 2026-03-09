"""
Retriever: high-level interface for chunk retrieval.

Wraps ChunkIndex and translates results into RetrievedChunk schema objects.
Supports persistence so the index only needs to be built once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from llm_evals_lab.retrieval.embedder import BaseEmbedder, get_embedder
from llm_evals_lab.retrieval.index import ChunkIndex
from llm_evals_lab.schemas import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """
    Top-level retrieval interface.

    Usage
    -----
    >>> retriever = Retriever.build(chunks, embedding_backend="tfidf")
    >>> results = retriever.retrieve("What is the refund policy?", top_k=5)
    """

    def __init__(self, index: ChunkIndex, top_k: int = 5, score_threshold: float = 0.0) -> None:
        self.index = index
        self.top_k = top_k
        self.score_threshold = score_threshold

    @classmethod
    def build(
        cls,
        chunks: list[DocumentChunk],
        embedding_backend: str = "tfidf",
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        score_threshold: float = 0.0,
        index_path: Optional[Path] = None,
    ) -> "Retriever":
        """
        Build a Retriever from a list of document chunks.

        If ``index_path`` is provided and the file exists, load from disk.
        Otherwise build from scratch and optionally save.
        """
        embedder = get_embedder(backend=embedding_backend, model_name=model_name)

        if index_path and index_path.exists():
            logger.info("Loading existing index from %s", index_path)
            index = ChunkIndex.load(index_path, embedder)
        else:
            index = ChunkIndex(embedder)
            index.build(chunks)
            if index_path:
                index.save(index_path)

        return cls(index, top_k=top_k, score_threshold=score_threshold)

    @classmethod
    def from_config(
        cls,
        chunks: list[DocumentChunk],
        cfg: dict,
        index_path: Optional[Path] = None,
    ) -> "Retriever":
        """Construct from a retrieval config dict (from configs/retrieval.yaml)."""
        ret_cfg = cfg.get("retrieval", {})
        tfidf_cfg = ret_cfg.get("tfidf", {})
        return cls.build(
            chunks=chunks,
            embedding_backend=ret_cfg.get("embedding_backend", "tfidf"),
            model_name=ret_cfg.get("sentence_transformer_model", "all-MiniLM-L6-v2"),
            top_k=ret_cfg.get("top_k", 5),
            score_threshold=ret_cfg.get("score_threshold", 0.0),
            index_path=index_path,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query : str
            The natural-language query.
        top_k : int, optional
            Override the default top_k.

        Returns
        -------
        list[RetrievedChunk]
            Ranked list of retrieved chunks with similarity scores.
        """
        k = top_k if top_k is not None else self.top_k
        raw_results = self.index.search(query, top_k=k, score_threshold=self.score_threshold)

        retrieved: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(raw_results, start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    chunk_text=chunk.chunk_text,
                    score=round(score, 4),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return retrieved

    @property
    def embedding_backend(self) -> str:
        return self.index.embedder.name
