"""
Document chunking strategies for the LLM Evals Lab.

Provides configurable chunking with sliding-window, sentence-boundary,
and fixed-size strategies. All strategies return DocumentChunk objects.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from llm_evals_lab.schemas import DocumentChunk, SourceDocument
from llm_evals_lab.utils import make_chunk_id

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["sliding_window", "sentence", "fixed"]


class Chunker:
    """
    Splits source documents into overlapping chunks.

    Parameters
    ----------
    strategy:
        Chunking strategy. "sliding_window" (default) uses word-level
        windows with overlap. "sentence" splits on sentence boundaries.
        "fixed" uses non-overlapping fixed-size windows.
    chunk_size:
        Target chunk size in words.
    chunk_overlap:
        Overlap in words between consecutive chunks (sliding_window only).
    min_chunk_size:
        Minimum words to keep a chunk.
    max_chunk_size:
        Hard cap on chunk size (words).
    """

    def __init__(
        self,
        strategy: ChunkStrategy = "sliding_window",
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        min_chunk_size: int = 30,
        max_chunk_size: int = 400,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    @classmethod
    def from_config(cls, cfg: dict) -> "Chunker":
        """Construct a Chunker from a retrieval config dict."""
        c = cfg.get("chunking", {})
        return cls(
            strategy=c.get("strategy", "sliding_window"),
            chunk_size=c.get("chunk_size", 200),
            chunk_overlap=c.get("chunk_overlap", 40),
            min_chunk_size=c.get("min_chunk_size", 30),
            max_chunk_size=c.get("max_chunk_size", 400),
        )

    def chunk_document(self, doc: SourceDocument) -> list[DocumentChunk]:
        """Split a single document into chunks."""
        if self.strategy == "sliding_window":
            texts = self._sliding_window(doc.full_text)
        elif self.strategy == "sentence":
            texts = self._sentence_split(doc.full_text)
        elif self.strategy == "fixed":
            texts = self._fixed_split(doc.full_text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        chunks: list[DocumentChunk] = []
        for i, text in enumerate(texts):
            wc = len(text.split())
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(doc.doc_id, i),
                    doc_id=doc.doc_id,
                    chunk_text=text,
                    chunk_order=i,
                    word_count=wc,
                    token_count_estimate=int(wc * 1.33),
                    metadata={
                        "title": doc.title,
                        "source_type": doc.source_type,
                        "category": doc.category,
                        **doc.metadata,
                    },
                )
            )
        return chunks

    def chunk_corpus(self, docs: list[SourceDocument]) -> list[DocumentChunk]:
        """Chunk an entire corpus, returning a flat list of DocumentChunks."""
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(
            "Chunked %d documents into %d chunks (strategy=%s)",
            len(docs),
            len(all_chunks),
            self.strategy,
        )
        return all_chunks

    # ── Chunking strategies ───────────────────────────────────────────────────

    def _sliding_window(self, text: str) -> list[str]:
        """Word-level sliding window with overlap."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text] if len(words) >= self.min_chunk_size else []

        stride = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            if len(chunk_words) >= self.min_chunk_size:
                chunk_text = " ".join(chunk_words[: self.max_chunk_size])
                chunks.append(chunk_text)
            start += stride
            if end == len(words):
                break
        return chunks

    def _sentence_split(self, text: str) -> list[str]:
        """
        Sentence-aware chunking: accumulate sentences up to chunk_size words,
        then start a new chunk. Cheap and dependency-free.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        current_words: list[str] = []

        for sent in sentences:
            sent_words = sent.split()
            if (
                current_words
                and len(current_words) + len(sent_words) > self.chunk_size
            ):
                joined = " ".join(current_words)
                if len(current_words) >= self.min_chunk_size:
                    chunks.append(joined)
                current_words = sent_words
            else:
                current_words.extend(sent_words)

        if len(current_words) >= self.min_chunk_size:
            chunks.append(" ".join(current_words))

        return chunks

    def _fixed_split(self, text: str) -> list[str]:
        """Non-overlapping fixed-size word chunks."""
        words = text.split()
        chunks: list[str] = []
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i : i + self.chunk_size]
            if len(chunk_words) >= self.min_chunk_size:
                chunks.append(" ".join(chunk_words))
        return chunks
