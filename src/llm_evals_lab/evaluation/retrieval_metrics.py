"""
Retrieval quality metrics for the LLM Evals Lab.

All metrics are interpretable and documented. Where standard definitions
apply (Hit@K, MRR, Precision, Recall), we follow the IR literature.

Metrics
-------
- hit_at_k: binary — was any expected doc retrieved in the top-K?
- reciprocal_rank: 1/rank of the first relevant doc (0 if none found)
- context_precision: fraction of retrieved chunks whose parent doc is relevant
- context_recall: fraction of expected docs covered by retrieved chunks
- retrieved_relevant_count: raw count of relevant docs in the retrieved set
"""

from __future__ import annotations

from llm_evals_lab.schemas import RetrievalMetrics, RetrievedChunk


def compute_retrieval_metrics(
    retrieved_chunks: list[RetrievedChunk],
    expected_doc_ids: list[str],
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single query.

    Parameters
    ----------
    retrieved_chunks : list[RetrievedChunk]
        Ranked list of retrieved chunks (rank 1 = most relevant).
    expected_doc_ids : list[str]
        Ground-truth doc IDs that contain the answer.

    Returns
    -------
    RetrievalMetrics
    """
    if not expected_doc_ids:
        # Unanswerable question — evaluate differently
        return _unanswerable_retrieval_metrics(retrieved_chunks)

    expected_set = set(expected_doc_ids)
    retrieved_doc_ids_ordered = [c.doc_id for c in retrieved_chunks]

    # ── Hit@K ─────────────────────────────────────────────────────────────────
    hit = float(any(did in expected_set for did in retrieved_doc_ids_ordered))

    # ── Reciprocal Rank ───────────────────────────────────────────────────────
    rr = 0.0
    for rank, did in enumerate(retrieved_doc_ids_ordered, start=1):
        if did in expected_set:
            rr = 1.0 / rank
            break

    # ── Context Precision ─────────────────────────────────────────────────────
    # Fraction of retrieved chunks whose doc is in the expected set
    if retrieved_chunks:
        relevant_chunks = sum(1 for c in retrieved_chunks if c.doc_id in expected_set)
        precision = relevant_chunks / len(retrieved_chunks)
    else:
        precision = 0.0

    # ── Context Recall ────────────────────────────────────────────────────────
    # Fraction of expected docs that appear in the retrieved set
    retrieved_doc_set = set(retrieved_doc_ids_ordered)
    covered = expected_set & retrieved_doc_set
    recall = len(covered) / len(expected_set)

    # ── Retrieved relevant count ──────────────────────────────────────────────
    relevant_count = len(covered)

    return RetrievalMetrics(
        hit_at_k=hit,
        reciprocal_rank=round(rr, 4),
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        retrieved_relevant_count=relevant_count,
        retrieved_total_count=len(retrieved_chunks),
    )


def _unanswerable_retrieval_metrics(
    retrieved_chunks: list[RetrievedChunk],
) -> RetrievalMetrics:
    """
    For unanswerable questions we cannot measure recall/hit.
    We return neutral values and flag via retrieved_relevant_count = -1.
    """
    return RetrievalMetrics(
        hit_at_k=0.0,
        reciprocal_rank=0.0,
        context_precision=0.0,
        context_recall=1.0,  # no expected docs, so recall is trivially satisfied
        retrieved_relevant_count=0,
        retrieved_total_count=len(retrieved_chunks),
    )
