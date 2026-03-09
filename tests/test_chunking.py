"""
Tests for the document chunking module.
"""

from __future__ import annotations

import pytest

from llm_evals_lab.data.chunking import Chunker
from llm_evals_lab.schemas import DocumentChunk, SourceDocument


@pytest.fixture
def long_doc() -> SourceDocument:
    """A document long enough to produce multiple chunks."""
    words = " ".join([f"word{i}" for i in range(500)])
    return SourceDocument(
        doc_id="doc_long",
        title="Long Document",
        source_type="guide",
        category="test",
        full_text=words,
    )


@pytest.fixture
def short_doc() -> SourceDocument:
    """A document shorter than min_chunk_size."""
    return SourceDocument(
        doc_id="doc_short",
        title="Short Document",
        source_type="guide",
        category="test",
        full_text="This is a very short document.",
    )


class TestSlidingWindowChunker:
    def test_produces_multiple_chunks_for_long_doc(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(long_doc)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(long_doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_reference_doc(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100)
        chunks = chunker.chunk_document(long_doc)
        for chunk in chunks:
            assert chunk.doc_id == long_doc.doc_id
            assert long_doc.doc_id in chunk.chunk_id

    def test_word_counts_populated(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100)
        chunks = chunker.chunk_document(long_doc)
        for chunk in chunks:
            assert chunk.word_count > 0
            assert chunk.word_count == len(chunk.chunk_text.split())

    def test_token_count_estimate(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100)
        chunks = chunker.chunk_document(long_doc)
        for chunk in chunks:
            # Token estimate should be roughly 1.33x word count
            expected = int(chunk.word_count * 1.33)
            assert chunk.token_count_estimate == expected

    def test_min_chunk_size_respected(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100, min_chunk_size=50)
        chunks = chunker.chunk_document(long_doc)
        for chunk in chunks:
            assert chunk.word_count >= 50

    def test_max_chunk_size_respected(self, long_doc):
        max_size = 150
        chunker = Chunker(strategy="sliding_window", chunk_size=max_size, max_chunk_size=max_size)
        chunks = chunker.chunk_document(long_doc)
        for chunk in chunks:
            assert chunk.word_count <= max_size + 5  # small tolerance

    def test_short_doc_produces_one_chunk(self, sample_documents):
        doc = sample_documents[0]
        chunker = Chunker(strategy="sliding_window", chunk_size=200, min_chunk_size=5)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_short_doc_below_min_chunk_size_produces_no_chunks(self, short_doc):
        chunker = Chunker(strategy="sliding_window", min_chunk_size=100)
        chunks = chunker.chunk_document(short_doc)
        # Short doc has 7 words, min_chunk_size=100 → should produce 0 chunks
        assert len(chunks) == 0

    def test_chunk_order_sequential(self, long_doc):
        chunker = Chunker(strategy="sliding_window", chunk_size=100)
        chunks = chunker.chunk_document(long_doc)
        orders = [c.chunk_order for c in chunks]
        assert orders == list(range(len(chunks)))


class TestSentenceChunker:
    def test_produces_chunks(self, sample_documents):
        doc = sample_documents[0]
        chunker = Chunker(strategy="sentence", chunk_size=50, min_chunk_size=5)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_text_is_non_empty(self, sample_documents):
        doc = sample_documents[0]
        chunker = Chunker(strategy="sentence", min_chunk_size=5)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.chunk_text.strip() != ""


class TestFixedChunker:
    def test_non_overlapping(self, long_doc):
        chunker = Chunker(strategy="fixed", chunk_size=100, min_chunk_size=10)
        chunks = chunker.chunk_document(long_doc)
        assert len(chunks) >= 1
        # Fixed chunks don't overlap: total words ≥ doc words
        total = sum(c.word_count for c in chunks)
        assert total <= long_doc.word_count


class TestCorpusChunking:
    def test_chunk_corpus_flat_list(self, sample_documents):
        chunker = Chunker(strategy="sliding_window", chunk_size=50, min_chunk_size=5)
        chunks = chunker.chunk_corpus(sample_documents)
        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_corpus_covers_all_docs(self, sample_documents):
        chunker = Chunker(strategy="sliding_window", chunk_size=50, min_chunk_size=5)
        chunks = chunker.chunk_corpus(sample_documents)
        doc_ids_in_chunks = {c.doc_id for c in chunks}
        doc_ids_in_corpus = {d.doc_id for d in sample_documents}
        assert doc_ids_in_corpus == doc_ids_in_chunks

    def test_metadata_propagated(self, sample_documents):
        chunker = Chunker(strategy="sliding_window", chunk_size=50, min_chunk_size=5)
        chunks = chunker.chunk_corpus(sample_documents)
        for chunk in chunks:
            assert "category" in chunk.metadata
