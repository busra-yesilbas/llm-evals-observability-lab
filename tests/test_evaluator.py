"""
Tests for the evaluation engine.
"""

from __future__ import annotations

import pytest

from llm_evals_lab.evaluation.answer_quality import (
    compute_abstention_quality,
    compute_answer_relevance,
    compute_citation_coverage,
    compute_exact_match_proxy,
    compute_faithfulness_proxy,
)
from llm_evals_lab.evaluation.groundedness import (
    compute_groundedness,
    compute_hallucination_risk,
)
from llm_evals_lab.evaluation.retrieval_metrics import compute_retrieval_metrics
from llm_evals_lab.schemas import (
    AnswerMetrics,
    Difficulty,
    EvalExample,
    FailureMode,
    GeneratedAnswer,
    PromptStrategy,
    QuestionCategory,
    RetrievedChunk,
    RunRecord,
)
from llm_evals_lab.evaluation.evaluator import Evaluator
from llm_evals_lab.utils import new_run_id


class TestRetrievalMetrics:
    def test_perfect_recall(self, sample_retrieved_chunks, sample_eval_examples):
        # sample_retrieved_chunks contains doc_test_001 and doc_test_002
        metrics = compute_retrieval_metrics(
            sample_retrieved_chunks, ["doc_test_001"]
        )
        assert metrics.hit_at_k == 1.0
        assert metrics.context_recall == 1.0
        assert metrics.reciprocal_rank == 1.0  # first chunk is from doc_test_001

    def test_no_relevant_docs_retrieved(self, sample_retrieved_chunks):
        metrics = compute_retrieval_metrics(
            sample_retrieved_chunks, ["doc_does_not_exist"]
        )
        assert metrics.hit_at_k == 0.0
        assert metrics.context_recall == 0.0
        assert metrics.reciprocal_rank == 0.0

    def test_no_expected_docs_unanswerable(self, sample_retrieved_chunks):
        # Unanswerable: expected_doc_ids is empty
        metrics = compute_retrieval_metrics(sample_retrieved_chunks, [])
        # Our convention: recall=1.0 (trivially satisfied) for unanswerable
        assert metrics.context_recall == 1.0

    def test_partial_recall(self, sample_retrieved_chunks):
        metrics = compute_retrieval_metrics(
            sample_retrieved_chunks, ["doc_test_001", "doc_test_003"]
        )
        # retrieved: [doc_test_001, doc_test_002] — doc_test_003 not in results
        assert 0 < metrics.context_recall < 1.0

    def test_scores_in_range(self, sample_retrieved_chunks):
        metrics = compute_retrieval_metrics(
            sample_retrieved_chunks, ["doc_test_001"]
        )
        assert 0.0 <= metrics.hit_at_k <= 1.0
        assert 0.0 <= metrics.reciprocal_rank <= 1.0
        assert 0.0 <= metrics.context_precision <= 1.0
        assert 0.0 <= metrics.context_recall <= 1.0


class TestGroundedness:
    def test_perfect_abstention_is_grounded(self):
        score = compute_groundedness("", [], abstained=True)
        assert score == 1.0

    def test_no_chunks_returns_zero(self):
        score = compute_groundedness("Some answer text", [])
        assert score == 0.0

    def test_high_overlap_gives_high_score(self, sample_retrieved_chunks):
        # Answer is a direct quote from the first chunk
        chunk = sample_retrieved_chunks[0]
        score = compute_groundedness(chunk.chunk_text[:100], sample_retrieved_chunks)
        assert score > 0.3

    def test_unrelated_answer_gives_low_score(self, sample_retrieved_chunks):
        score = compute_groundedness(
            "The weather today is sunny and warm in Paris.",
            sample_retrieved_chunks,
        )
        assert score < 0.4

    def test_score_in_range(self, sample_retrieved_chunks):
        score = compute_groundedness("some answer about refund policy", sample_retrieved_chunks)
        assert 0.0 <= score <= 1.0


class TestHallucinationRisk:
    def test_abstention_zero_risk(self, sample_retrieved_chunks):
        risk = compute_hallucination_risk("", sample_retrieved_chunks, 1.0, abstained=True)
        assert risk == 0.0

    def test_no_chunks_max_risk(self):
        risk = compute_hallucination_risk("Some fabricated text.", [], 0.0)
        assert risk == 1.0

    def test_risk_inverse_of_groundedness(self, sample_retrieved_chunks):
        chunk = sample_retrieved_chunks[0]
        gs = compute_groundedness(chunk.chunk_text[:100], sample_retrieved_chunks)
        risk = compute_hallucination_risk(chunk.chunk_text[:100], sample_retrieved_chunks, gs)
        # Risk should be 1 - groundedness (approximately)
        assert risk < 0.7


class TestAnswerQuality:
    def test_answer_relevance_high_for_good_answer(self):
        score = compute_answer_relevance(
            "What is the refund policy?",
            "NovaSaaS offers a 30-day money-back guarantee for refund requests.",
        )
        assert score > 0.2

    def test_answer_relevance_low_for_unrelated(self):
        # Use vocabulary with zero overlap (no shared stopwords either)
        score = compute_answer_relevance(
            "refund billing subscription cancellation",
            "astronomy telescope galaxy nebula constellation",
        )
        assert score < 0.3

    def test_abstained_relevance_is_neutral(self):
        score = compute_answer_relevance("any question", "", abstained=True)
        assert score == 0.5

    def test_citation_coverage_perfect(self):
        score = compute_citation_coverage(["doc_test_001"], ["doc_test_001"])
        assert score == 1.0

    def test_citation_coverage_zero(self):
        score = compute_citation_coverage([], ["doc_test_001"])
        assert score == 0.0

    def test_citation_coverage_partial(self):
        score = compute_citation_coverage(["doc_test_001"], ["doc_test_001", "doc_test_002"])
        assert 0.0 < score < 1.0

    def test_citation_coverage_no_expected(self):
        score = compute_citation_coverage(["doc_test_001"], [])
        assert score == 1.0

    def test_exact_match_proxy_perfect(self):
        text = "NovaSaaS offers a 30-day money-back guarantee"
        score = compute_exact_match_proxy(text, text)
        assert score == 1.0

    def test_exact_match_proxy_zero_for_empty(self):
        score = compute_exact_match_proxy("", "some reference")
        assert score == 0.0

    def test_abstention_quality_correct_abstention(self):
        # Unanswerable + abstained → 1.0
        score = compute_abstention_quality(abstained=True, is_answerable=False, answer_relevance=0.5)
        assert score == 1.0

    def test_abstention_quality_incorrect_abstention(self):
        # Answerable + abstained → 0.0
        score = compute_abstention_quality(abstained=True, is_answerable=True, answer_relevance=0.8)
        assert score == 0.0

    def test_abstention_quality_answered_unanswerable(self):
        # Unanswerable + answered → 0.0
        score = compute_abstention_quality(abstained=False, is_answerable=False, answer_relevance=0.6)
        assert score == 0.0


class TestEvaluator:
    def _make_run_record(self, answer_text: str, retrieved_chunks, abstained: bool = False) -> RunRecord:
        return RunRecord(
            run_id=new_run_id(),
            experiment_id="test",
            query="What is the refund window?",
            example_id="test_eval_001",
            prompt_strategy=PromptStrategy.BASELINE,
            top_k=3,
            embedding_backend="tfidf",
            retrieved_chunks=retrieved_chunks,
            retrieved_doc_ids=[c.doc_id for c in retrieved_chunks],
            prompt_text="test prompt",
            generated_answer=GeneratedAnswer(
                answer_text=answer_text,
                cited_chunk_ids=[retrieved_chunks[0].chunk_id] if retrieved_chunks else [],
                cited_doc_ids=[retrieved_chunks[0].doc_id] if retrieved_chunks else [],
                abstained=abstained,
                prompt_strategy=PromptStrategy.BASELINE,
            ),
        )

    def test_evaluate_returns_run_metrics(self, sample_retrieved_chunks, sample_eval_examples):
        evaluator = Evaluator()
        record = self._make_run_record(
            "NovaSaaS offers a 30-day money-back guarantee",
            sample_retrieved_chunks,
        )
        metrics = evaluator.evaluate(record, sample_eval_examples[0])
        assert metrics is not None
        assert 0.0 <= metrics.answer.overall_score <= 1.0

    def test_metrics_all_in_range(self, sample_retrieved_chunks, sample_eval_examples):
        evaluator = Evaluator()
        record = self._make_run_record(
            "The refund policy allows 30 days for requests.",
            sample_retrieved_chunks,
        )
        metrics = evaluator.evaluate(record, sample_eval_examples[0])
        m = metrics.answer
        for field in [
            m.answer_relevance_score, m.groundedness_score, m.citation_coverage_score,
            m.exact_match_proxy, m.faithfulness_proxy, m.hallucination_risk_score,
            m.overall_score,
        ]:
            assert 0.0 <= field <= 1.0

    def test_correct_abstention_scores_high(self, sample_retrieved_chunks, sample_eval_examples):
        evaluator = Evaluator()
        unanswerable_example = sample_eval_examples[2]  # test_eval_003
        record = self._make_run_record(
            "I cannot answer this question.",
            sample_retrieved_chunks,
            abstained=True,
        )
        metrics = evaluator.evaluate(record, unanswerable_example)
        assert metrics.answer.abstention_quality_score == 1.0

    def test_failure_modes_detected(self, sample_retrieved_chunks, sample_eval_examples):
        evaluator = Evaluator()
        # Empty answer on answerable question → incomplete_answer
        record = self._make_run_record("", sample_retrieved_chunks)
        metrics = evaluator.evaluate(record, sample_eval_examples[0])
        modes = [fm.value for fm in metrics.failure_modes]
        assert FailureMode.NONE.value not in modes or "incomplete_answer" in modes

    def test_zero_metrics_for_no_answer(self, sample_retrieved_chunks, sample_eval_examples):
        evaluator = Evaluator()
        record = RunRecord(
            run_id=new_run_id(),
            experiment_id="test",
            query="test",
            retrieved_chunks=sample_retrieved_chunks,
            retrieved_doc_ids=[],
            generated_answer=None,
        )
        metrics = evaluator._zero_metrics(record, sample_eval_examples[0])
        assert metrics.answer.overall_score == 0.0
