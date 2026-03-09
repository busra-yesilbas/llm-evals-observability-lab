"""
Composite metric computation and failure mode detection.

This module assembles individual metric scores into RunMetrics and
identifies failure modes based on configurable thresholds.
"""

from __future__ import annotations

import logging

from llm_evals_lab.schemas import (
    AnswerMetrics,
    EvalExample,
    FailureMode,
    GeneratedAnswer,
    RetrievalMetrics,
    RetrievedChunk,
    RunMetrics,
)
from llm_evals_lab.utils import estimate_cost

logger = logging.getLogger(__name__)

# Default score weights for overall_score
_DEFAULT_WEIGHTS = {
    "answer_relevance": 0.20,
    "groundedness": 0.25,
    "citation_coverage": 0.15,
    "context_precision": 0.15,
    "context_recall": 0.15,
    "faithfulness_proxy": 0.10,
}


def compute_overall_score(
    answer_metrics: AnswerMetrics,
    retrieval_metrics: RetrievalMetrics,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute weighted composite score from answer and retrieval metrics.

    Weights are configurable via eval.yaml. Scores are clamped to [0, 1].
    """
    w = weights or _DEFAULT_WEIGHTS

    score = (
        w.get("answer_relevance", 0.20) * answer_metrics.answer_relevance_score
        + w.get("groundedness", 0.25) * answer_metrics.groundedness_score
        + w.get("citation_coverage", 0.15) * answer_metrics.citation_coverage_score
        + w.get("context_precision", 0.15) * retrieval_metrics.context_precision
        + w.get("context_recall", 0.15) * retrieval_metrics.context_recall
        + w.get("faithfulness_proxy", 0.10) * answer_metrics.faithfulness_proxy
    )
    return round(min(1.0, max(0.0, score)), 4)


def detect_failure_modes(
    answer_metrics: AnswerMetrics,
    retrieval_metrics: RetrievalMetrics,
    generated_answer: GeneratedAnswer,
    eval_example: EvalExample,
    thresholds: dict | None = None,
) -> list[FailureMode]:
    """
    Identify which failure modes apply to this run.

    Failure modes are not mutually exclusive — a single run can exhibit
    multiple failure patterns.

    Parameters
    ----------
    thresholds : dict, optional
        Override default thresholds from eval.yaml.

    Returns
    -------
    list[FailureMode]
    """
    t = thresholds or {}
    failures: list[FailureMode] = []

    # Unsupported answer: low groundedness on an answerable question
    if (
        not generated_answer.abstained
        and eval_example.is_answerable
        and answer_metrics.groundedness_score
        < t.get("unsupported_answer_threshold", 0.30)
    ):
        failures.append(FailureMode.UNSUPPORTED_ANSWER)

    # Weak retrieval: low context recall
    if retrieval_metrics.context_recall < t.get("weak_retrieval_threshold", 0.30):
        failures.append(FailureMode.WEAK_RETRIEVAL)

    # Missing citation: answerable but no citations
    if (
        eval_example.is_answerable
        and not generated_answer.abstained
        and answer_metrics.citation_coverage_score
        < t.get("missing_citation_threshold", 0.20)
    ):
        failures.append(FailureMode.MISSING_CITATION)

    # Incorrect abstention: abstained on answerable question
    if generated_answer.abstained and eval_example.is_answerable:
        failures.append(FailureMode.INCORRECT_ABSTENTION)

    # Overconfident answer: answered an unanswerable question
    if not generated_answer.abstained and not eval_example.is_answerable:
        failures.append(FailureMode.OVERCONFIDENT_ANSWER)

    # Incomplete answer: very short answer on answerable question
    min_words = t.get("incomplete_answer_min_words", 10)
    if (
        eval_example.is_answerable
        and not generated_answer.abstained
        and generated_answer.word_count < min_words
    ):
        failures.append(FailureMode.INCOMPLETE_ANSWER)

    if not failures:
        failures.append(FailureMode.NONE)

    return failures


def assemble_run_metrics(
    answer_metrics: AnswerMetrics,
    retrieval_metrics: RetrievalMetrics,
    generated_answer: GeneratedAnswer,
    eval_example: EvalExample,
    latency_ms: float,
    prompt_word_count: int,
    eval_cfg: dict | None = None,
) -> RunMetrics:
    """
    Assemble the final RunMetrics object from component scores.

    Parameters
    ----------
    prompt_word_count : int
        Word count of the full prompt (used for cost estimation).
    eval_cfg : dict, optional
        eval section of LabConfig (for thresholds, weights, cost).
    """
    cfg = eval_cfg or {}
    cost_cfg = cfg.get("cost", {})
    threshold_cfg = cfg.get("thresholds", {})
    failure_cfg = cfg.get("failure_modes", {})

    # Recompute overall_score with config weights
    weights = cfg.get("score_weights")
    overall = compute_overall_score(answer_metrics, retrieval_metrics, weights)

    # Create updated AnswerMetrics with overall_score filled in
    answer_metrics_with_overall = AnswerMetrics(
        answer_relevance_score=answer_metrics.answer_relevance_score,
        groundedness_score=answer_metrics.groundedness_score,
        citation_coverage_score=answer_metrics.citation_coverage_score,
        exact_match_proxy=answer_metrics.exact_match_proxy,
        faithfulness_proxy=answer_metrics.faithfulness_proxy,
        hallucination_risk_score=answer_metrics.hallucination_risk_score,
        abstention_quality_score=answer_metrics.abstention_quality_score,
        overall_score=overall,
    )

    failure_modes = detect_failure_modes(
        answer_metrics=answer_metrics_with_overall,
        retrieval_metrics=retrieval_metrics,
        generated_answer=generated_answer,
        eval_example=eval_example,
        thresholds=failure_cfg,
    )

    prompt_tokens = int(prompt_word_count * cost_cfg.get("words_to_tokens_ratio", 1.33))
    output_tokens = generated_answer.token_count_estimate
    cost = estimate_cost(
        prompt_words=prompt_word_count,
        output_words=generated_answer.word_count,
        input_cost_per_1k=cost_cfg.get("input_cost_per_1k", 0.0005),
        output_cost_per_1k=cost_cfg.get("output_cost_per_1k", 0.0015),
        words_to_tokens=cost_cfg.get("words_to_tokens_ratio", 1.33),
    )

    warnings: list[str] = []
    if answer_metrics_with_overall.hallucination_risk_score > threshold_cfg.get(
        "hallucination_risk_warn", 0.6
    ):
        warnings.append(
            f"High hallucination risk: {answer_metrics_with_overall.hallucination_risk_score:.2f}"
        )

    return RunMetrics(
        retrieval=retrieval_metrics,
        answer=answer_metrics_with_overall,
        latency_ms=round(latency_ms, 2),
        prompt_tokens_estimate=prompt_tokens,
        output_tokens_estimate=output_tokens,
        estimated_cost_usd=cost,
        failure_modes=failure_modes,
        warnings=warnings,
    )
