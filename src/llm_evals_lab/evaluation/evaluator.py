"""
Evaluator: orchestrates all metric computations for a single pipeline run.
"""

from __future__ import annotations

import logging

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
from llm_evals_lab.evaluation.metrics import assemble_run_metrics
from llm_evals_lab.evaluation.retrieval_metrics import compute_retrieval_metrics
from llm_evals_lab.schemas import AnswerMetrics, EvalExample, RunMetrics, RunRecord

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a completed RunRecord against a ground-truth EvalExample.

    Computes all retrieval and answer quality metrics, detects failure modes,
    and returns a RunMetrics object.

    Parameters
    ----------
    eval_cfg : dict
        The 'eval' section from LabConfig (thresholds, weights, cost params).
    """

    def __init__(self, eval_cfg: dict | None = None) -> None:
        self.eval_cfg = eval_cfg or {}

    def evaluate(self, record: RunRecord, example: EvalExample) -> RunMetrics:
        """
        Run all metric computations for a single pipeline run.

        Parameters
        ----------
        record : RunRecord
            Completed pipeline run (must have generated_answer).
        example : EvalExample
            Ground-truth reference.

        Returns
        -------
        RunMetrics
        """
        answer = record.generated_answer
        if answer is None:
            logger.warning("RunRecord %s has no generated answer — returning zero metrics", record.run_id)
            return self._zero_metrics(record, example)

        chunks = record.retrieved_chunks

        # ── Retrieval metrics ─────────────────────────────────────────────────
        retrieval_metrics = compute_retrieval_metrics(chunks, example.expected_doc_ids)

        # ── Groundedness and hallucination ────────────────────────────────────
        groundedness = compute_groundedness(answer.answer_text, chunks, answer.abstained)
        hallucination_risk = compute_hallucination_risk(
            answer.answer_text, chunks, groundedness, answer.abstained
        )

        # ── Answer quality ────────────────────────────────────────────────────
        answer_relevance = compute_answer_relevance(
            record.query, answer.answer_text, answer.abstained
        )
        citation_coverage = compute_citation_coverage(
            answer.cited_doc_ids, example.expected_doc_ids
        )
        exact_match = compute_exact_match_proxy(
            answer.answer_text, example.reference_answer, answer.abstained
        )
        faithfulness = compute_faithfulness_proxy(
            answer.answer_text, chunks, answer.abstained
        )
        abstention_quality = compute_abstention_quality(
            answer.abstained, example.is_answerable, answer_relevance
        )

        answer_metrics = AnswerMetrics(
            answer_relevance_score=answer_relevance,
            groundedness_score=groundedness,
            citation_coverage_score=citation_coverage,
            exact_match_proxy=exact_match,
            faithfulness_proxy=faithfulness,
            hallucination_risk_score=hallucination_risk,
            abstention_quality_score=abstention_quality,
            overall_score=0.0,  # will be recomputed in assemble_run_metrics
        )

        # ── Assemble final RunMetrics ─────────────────────────────────────────
        prompt_word_count = len(record.prompt_text.split())
        latency_ms = record.metrics.latency_ms if record.metrics else 0.0

        run_metrics = assemble_run_metrics(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            generated_answer=answer,
            eval_example=example,
            latency_ms=latency_ms,
            prompt_word_count=prompt_word_count,
            eval_cfg=self.eval_cfg,
        )

        logger.debug(
            "Eval %s: overall=%.3f groundedness=%.3f retrieval_recall=%.3f",
            example.example_id,
            run_metrics.answer.overall_score,
            groundedness,
            retrieval_metrics.context_recall,
        )

        return run_metrics

    def _zero_metrics(self, record: RunRecord, example: EvalExample) -> RunMetrics:
        """Return a zeroed RunMetrics for runs that errored out."""
        from llm_evals_lab.schemas import FailureMode, RetrievalMetrics

        return RunMetrics(
            retrieval=RetrievalMetrics(
                hit_at_k=0.0,
                reciprocal_rank=0.0,
                context_precision=0.0,
                context_recall=0.0,
                retrieved_relevant_count=0,
                retrieved_total_count=len(record.retrieved_chunks),
            ),
            answer=AnswerMetrics(
                answer_relevance_score=0.0,
                groundedness_score=0.0,
                citation_coverage_score=0.0,
                exact_match_proxy=0.0,
                faithfulness_proxy=0.0,
                hallucination_risk_score=1.0,
                abstention_quality_score=0.0,
                overall_score=0.0,
            ),
            latency_ms=record.metrics.latency_ms if record.metrics else 0.0,
            prompt_tokens_estimate=0,
            output_tokens_estimate=0,
            estimated_cost_usd=0.0,
            failure_modes=[FailureMode.NONE],
            warnings=["No answer generated"],
        )
