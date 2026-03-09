"""
Groundedness evaluation for the LLM Evals Lab.

Groundedness measures how well the generated answer is supported by the
retrieved context — as opposed to being fabricated or hallucinated.

Implementation
--------------
We use two complementary signals:

1. **Chunk overlap score**: average word-level Jaccard overlap between the
   answer and each retrieved chunk. Higher → answer draws from retrieved text.

2. **Sentence attribution**: fraction of answer sentences that can be
   attributed to at least one retrieved chunk (overlap > threshold).

Both signals are averaged into a single groundedness_score ∈ [0, 1].

Limitations
-----------
These are heuristic proxies. A production system would use an NLI model or
LLM-as-a-judge for more accurate entailment scoring. We document this
explicitly so readers understand the tradeoff between rigor and API-cost.
"""

from __future__ import annotations

import re

from llm_evals_lab.schemas import RetrievedChunk
from llm_evals_lab.utils import normalize_text, word_overlap_score

# Sentence-level overlap threshold for attribution
_ATTRIBUTION_THRESHOLD = 0.08


def compute_groundedness(
    answer_text: str,
    retrieved_chunks: list[RetrievedChunk],
    abstained: bool = False,
) -> float:
    """
    Compute groundedness of the answer with respect to retrieved chunks.

    Parameters
    ----------
    answer_text : str
        The generated answer.
    retrieved_chunks : list[RetrievedChunk]
        Chunks that were provided as context.
    abstained : bool
        If True, the generator abstained; groundedness is set to 1.0
        (abstention is trivially grounded — it makes no unsupported claims).

    Returns
    -------
    float
        Groundedness score in [0, 1].
    """
    if abstained:
        return 1.0

    if not retrieved_chunks or not answer_text.strip():
        return 0.0

    chunk_texts = [c.chunk_text for c in retrieved_chunks]

    # Signal 1: mean chunk overlap
    overlaps = [word_overlap_score(answer_text, ct) for ct in chunk_texts]
    mean_overlap = sum(overlaps) / len(overlaps)
    max_overlap = max(overlaps)
    chunk_overlap_score = 0.4 * mean_overlap + 0.6 * max_overlap

    # Signal 2: sentence-level attribution
    sentences = _split_sentences(answer_text)
    if not sentences:
        return min(1.0, chunk_overlap_score * 5)

    attributed = 0
    for sent in sentences:
        best_sent_overlap = max(
            word_overlap_score(sent, ct) for ct in chunk_texts
        )
        if best_sent_overlap >= _ATTRIBUTION_THRESHOLD:
            attributed += 1

    attribution_score = attributed / len(sentences)

    # Combine: weight attribution more heavily
    groundedness = 0.35 * chunk_overlap_score + 0.65 * attribution_score
    return round(min(1.0, groundedness), 4)


def compute_hallucination_risk(
    answer_text: str,
    retrieved_chunks: list[RetrievedChunk],
    groundedness_score: float,
    abstained: bool = False,
) -> float:
    """
    Estimate hallucination risk as 1 - grounded_evidence_strength.

    Parameters
    ----------
    answer_text : str
        Generated answer.
    retrieved_chunks : list[RetrievedChunk]
        Retrieved context.
    groundedness_score : float
        Pre-computed groundedness score.
    abstained : bool
        If True, risk is 0.0.

    Returns
    -------
    float
        Hallucination risk in [0, 1]. Higher → more risk.
    """
    if abstained:
        return 0.0

    if not retrieved_chunks:
        return 1.0

    # Base risk is inverse of groundedness
    base_risk = 1.0 - groundedness_score

    # Penalty for answers that are longer than the total context (over-generation)
    context_words = sum(len(c.chunk_text.split()) for c in retrieved_chunks)
    answer_words = len(answer_text.split())
    length_penalty = 0.0
    if context_words > 0 and answer_words > context_words * 1.5:
        length_penalty = min(0.2, (answer_words / context_words - 1.5) * 0.1)

    risk = min(1.0, base_risk + length_penalty)
    return round(risk, 4)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (lightweight, no deps)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 4]
