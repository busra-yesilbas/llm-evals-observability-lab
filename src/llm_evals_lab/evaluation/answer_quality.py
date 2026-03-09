"""
Answer quality metrics for the LLM Evals Lab.

Metrics computed here assess the answer relative to the reference answer
and the retrieved context:

- answer_relevance_score: How relevant is the answer to the question?
- citation_coverage_score: What fraction of expected docs were cited?
- exact_match_proxy: Token-level F1 against reference answer.
- faithfulness_proxy: Does the answer contradict the context?
- abstention_quality_score: Did the system correctly abstain (or not)?

All metrics are [0, 1] scalars. They are heuristic proxies designed to
be interpretable without an LLM judge or NLI model.
"""

from __future__ import annotations

from llm_evals_lab.schemas import EvalExample, GeneratedAnswer, RetrievedChunk
from llm_evals_lab.utils import (
    keyword_coverage,
    normalize_text,
    token_overlap_f1,
    word_overlap_score,
)


def compute_answer_relevance(
    question: str,
    answer_text: str,
    abstained: bool = False,
) -> float:
    """
    Estimate how relevant the answer is to the question.

    Uses word-overlap between question and answer as a proxy for topical
    alignment. Abstained answers receive a moderate score since abstention
    may be the correct response.

    Returns float in [0, 1].
    """
    if abstained:
        return 0.5  # Neutral — could be correct or incorrect abstention

    if not answer_text.strip() or not question.strip():
        return 0.0

    overlap = word_overlap_score(question, answer_text)
    # Scale: raw overlap is typically 0.05–0.30; normalise to [0, 1]
    # We use a non-linear stretch to reward moderate-high overlap
    relevance = min(1.0, overlap * 4.0)
    return round(relevance, 4)


def compute_citation_coverage(
    cited_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float:
    """
    Fraction of expected document IDs that were cited in the answer.

    Returns 1.0 if no expected docs (nothing to cite).
    Returns float in [0, 1].
    """
    if not expected_doc_ids:
        return 1.0
    if not cited_doc_ids:
        return 0.0
    covered = set(cited_doc_ids) & set(expected_doc_ids)
    return round(len(covered) / len(expected_doc_ids), 4)


def compute_exact_match_proxy(
    answer_text: str,
    reference_answer: str,
    abstained: bool = False,
) -> float:
    """
    Token-level F1 between the generated answer and the reference answer.

    This is the standard SQuAD-style metric. It rewards partial overlap and
    is more informative than strict exact match for free-text answers.

    Returns float in [0, 1].
    """
    if abstained:
        return 0.0
    return round(token_overlap_f1(answer_text, reference_answer), 4)


def compute_faithfulness_proxy(
    answer_text: str,
    retrieved_chunks: list[RetrievedChunk],
    abstained: bool = False,
) -> float:
    """
    Proxy for faithfulness: does each sentence in the answer have
    lexical support in the retrieved context?

    Higher score → answer stays closer to retrieved text.
    This is a simplified proxy for NLI-based faithfulness.

    Returns float in [0, 1].
    """
    if abstained:
        return 1.0

    if not retrieved_chunks or not answer_text.strip():
        return 0.0

    combined_context = " ".join(c.chunk_text for c in retrieved_chunks)
    overlap = word_overlap_score(answer_text, combined_context)
    faithfulness = min(1.0, overlap * 3.5)
    return round(faithfulness, 4)


def compute_abstention_quality(
    abstained: bool,
    is_answerable: bool,
    answer_relevance: float,
) -> float:
    """
    Evaluate the quality of the abstention decision.

    - If the question is unanswerable and the system abstained → 1.0
    - If the question is unanswerable and the system answered → 0.0
    - If the question is answerable and the system abstained → 0.0
    - If the question is answerable and the system answered → use answer relevance

    Returns float in [0, 1].
    """
    if not is_answerable:
        return 1.0 if abstained else 0.0
    else:
        if abstained:
            return 0.0  # Should have answered
        return answer_relevance


def compute_key_point_coverage(
    answer_text: str,
    expected_key_points: list[str],
    abstained: bool = False,
) -> float:
    """
    Fraction of expected key points (factual elements) present in the answer.

    Returns float in [0, 1].
    """
    if abstained:
        return 0.0
    return round(keyword_coverage(answer_text, expected_key_points), 4)
