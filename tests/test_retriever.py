"""
Tests for the retrieval layer (embedder, index, retriever).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_evals_lab.retrieval.embedder import TFIDFEmbedder, get_embedder
from llm_evals_lab.retrieval.index import ChunkIndex
from llm_evals_lab.retrieval.retriever import Retriever
from llm_evals_lab.schemas import RetrievedChunk


class TestTFIDFEmbedder:
    def test_fit_and_embed(self, sample_chunks):
        texts = [c.chunk_text for c in sample_chunks]
        embedder = TFIDFEmbedder()
        embedder.fit(texts)
        vecs = embedder.embed(texts)
        assert vecs.shape == (len(texts), vecs.shape[1])

    def test_embed_returns_float32(self, sample_chunks):
        texts = [c.chunk_text for c in sample_chunks]
        embedder = TFIDFEmbedder()
        embedder.fit(texts)
        vecs = embedder.embed(texts)
        assert vecs.dtype == np.float32

    def test_embed_query_shape(self, sample_chunks):
        texts = [c.chunk_text for c in sample_chunks]
        embedder = TFIDFEmbedder()
        embedder.fit(texts)
        q_vec = embedder.embed_query("test query about refunds")
        assert q_vec.ndim == 1

    def test_embed_without_fit_raises(self):
        embedder = TFIDFEmbedder()
        with pytest.raises(RuntimeError):
            embedder.embed(["some text"])

    def test_l2_normalised(self, sample_chunks):
        texts = [c.chunk_text for c in sample_chunks]
        embedder = TFIDFEmbedder()
        embedder.fit(texts)
        vecs = embedder.embed(texts)
        norms = np.linalg.norm(vecs, axis=1)
        # Non-zero rows should have norm ≈ 1
        nonzero = norms > 1e-6
        if nonzero.any():
            np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-5)

    def test_name_property(self):
        embedder = TFIDFEmbedder()
        assert embedder.name == "tfidf"

    def test_save_and_load(self, sample_chunks, tmp_dir):
        texts = [c.chunk_text for c in sample_chunks]
        embedder = TFIDFEmbedder()
        embedder.fit(texts)
        path = tmp_dir / "embedder.pkl"
        embedder.save(path)
        loaded = TFIDFEmbedder.load(path)
        vecs_original = embedder.embed(texts)
        vecs_loaded = loaded.embed(texts)
        np.testing.assert_array_almost_equal(vecs_original, vecs_loaded)


class TestGetEmbedder:
    def test_tfidf_backend(self):
        embedder = get_embedder(backend="tfidf")
        assert embedder.name == "tfidf"

    def test_unknown_backend_falls_back_to_tfidf(self):
        # sentence-transformers likely not installed in test env; should fall back
        embedder = get_embedder(backend="sentence-transformers")
        assert "tfidf" in embedder.name.lower() or "sentence" in embedder.name.lower()


class TestChunkIndex:
    def test_build_and_search(self, sample_chunks):
        embedder = TFIDFEmbedder()
        index = ChunkIndex(embedder)
        index.build(sample_chunks)
        results = index.search("refund policy", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(c, type(sample_chunks[0])) for c, _ in results)

    def test_scores_in_range(self, sample_chunks):
        embedder = TFIDFEmbedder()
        index = ChunkIndex(embedder)
        index.build(sample_chunks)
        results = index.search("billing refund", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_results_sorted_descending(self, sample_chunks):
        embedder = TFIDFEmbedder()
        index = ChunkIndex(embedder)
        index.build(sample_chunks)
        results = index.search("SSO login security", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_before_build_raises(self, sample_chunks):
        embedder = TFIDFEmbedder()
        index = ChunkIndex(embedder)
        with pytest.raises(RuntimeError):
            index.search("test query")

    def test_num_chunks_property(self, sample_chunks):
        embedder = TFIDFEmbedder()
        index = ChunkIndex(embedder)
        index.build(sample_chunks)
        assert index.num_chunks == len(sample_chunks)

    def test_save_and_load(self, sample_chunks, tmp_dir):
        embedder = TFIDFEmbedder()
        index = ChunkIndex(embedder)
        index.build(sample_chunks)
        path = tmp_dir / "index.pkl"
        index.save(path)

        loaded_embedder = TFIDFEmbedder()
        loaded_index = ChunkIndex.load(path, loaded_embedder)
        assert loaded_index.num_chunks == len(sample_chunks)

        results_original = index.search("refund", top_k=3)
        results_loaded = loaded_index.search("refund", top_k=3)
        assert len(results_original) == len(results_loaded)


class TestRetriever:
    def test_build_and_retrieve(self, sample_chunks):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=3)
        results = retriever.retrieve("What is the refund policy?")
        assert len(results) <= 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_ranks_from_1(self, sample_chunks):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=5)
        results = retriever.retrieve("pricing plans seats")
        if results:
            assert results[0].rank == 1

    def test_retrieve_scores_in_range(self, sample_chunks):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf")
        results = retriever.retrieve("test query")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_top_k_override(self, sample_chunks):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=5)
        results = retriever.retrieve("billing", top_k=2)
        assert len(results) <= 2

    def test_relevant_doc_retrieved_for_clear_query(self, sample_chunks):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=5)
        results = retriever.retrieve("30-day money-back refund billing")
        retrieved_docs = [r.doc_id for r in results]
        assert "doc_test_001" in retrieved_docs

    def test_embedding_backend_property(self, sample_chunks):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf")
        assert retriever.embedding_backend == "tfidf"
