"""
Corpus loader for the LLM Evals Lab.

Loads source documents from JSON files in data/raw/ and provides
access to the processed chunk index from data/processed/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from llm_evals_lab.schemas import DocumentChunk, SourceDocument

logger = logging.getLogger(__name__)


class CorpusLoader:
    """
    Loads the document corpus and chunk index from disk.

    Supports both the raw source documents (data/raw/corpus.json)
    and the processed chunks (data/processed/chunks.json).
    """

    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    # ── Document loading ──────────────────────────────────────────────────────

    def load_documents(self, filename: str = "corpus.json") -> list[SourceDocument]:
        """Load source documents from the raw corpus JSON file."""
        path = self.raw_dir / filename
        if not path.exists():
            logger.warning("Corpus file not found: %s", path)
            return []

        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)

        docs = [SourceDocument(**r) for r in records]
        logger.info("Loaded %d source documents from %s", len(docs), path)
        return docs

    def load_chunks(self, filename: str = "chunks.json") -> list[DocumentChunk]:
        """Load processed chunks from disk."""
        path = self.processed_dir / filename
        if not path.exists():
            logger.warning("Chunks file not found: %s", path)
            return []

        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)

        chunks = [DocumentChunk(**r) for r in records]
        logger.info("Loaded %d chunks from %s", len(chunks), path)
        return chunks

    def save_documents(
        self, docs: list[SourceDocument], filename: str = "corpus.json"
    ) -> Path:
        """Persist source documents to disk."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump([d.model_dump() for d in docs], fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d documents to %s", len(docs), path)
        return path

    def save_chunks(
        self, chunks: list[DocumentChunk], filename: str = "chunks.json"
    ) -> Path:
        """Persist processed chunks to disk."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        path = self.processed_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump([c.model_dump() for c in chunks], fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d chunks to %s", len(chunks), path)
        return path

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def build_doc_index(
        self, docs: list[SourceDocument]
    ) -> dict[str, SourceDocument]:
        """Build a {doc_id: SourceDocument} lookup dict."""
        return {d.doc_id: d for d in docs}

    def build_chunk_index(
        self, chunks: list[DocumentChunk]
    ) -> dict[str, DocumentChunk]:
        """Build a {chunk_id: DocumentChunk} lookup dict."""
        return {c.chunk_id: c for c in chunks}

    def chunks_for_doc(
        self, chunks: list[DocumentChunk], doc_id: str
    ) -> list[DocumentChunk]:
        """Return all chunks belonging to a given document, in order."""
        return sorted(
            [c for c in chunks if c.doc_id == doc_id], key=lambda c: c.chunk_order
        )

    # ── Corpus statistics ─────────────────────────────────────────────────────

    @staticmethod
    def corpus_stats(
        docs: list[SourceDocument], chunks: list[DocumentChunk]
    ) -> dict:
        """Return a summary statistics dict for the corpus."""
        if not docs:
            return {}
        total_words = sum(d.word_count for d in docs)
        chunk_words = [c.word_count for c in chunks]
        categories: dict[str, int] = {}
        for d in docs:
            categories[d.category] = categories.get(d.category, 0) + 1

        return {
            "num_documents": len(docs),
            "num_chunks": len(chunks),
            "total_words": total_words,
            "avg_doc_words": round(total_words / len(docs), 1),
            "avg_chunk_words": round(sum(chunk_words) / len(chunks), 1) if chunks else 0,
            "categories": categories,
            "source_types": list({d.source_type for d in docs}),
        }
