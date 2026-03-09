"""
Pydantic schemas for all data structures in the LLM Evals Lab.

These are the canonical data contracts used throughout the project:
- corpus documents and chunks
- eval examples
- RAG pipeline outputs
- evaluation metrics
- trace records
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionCategory(str, Enum):
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    POLICY = "policy"
    COMPARISON = "comparison"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class FailureMode(str, Enum):
    UNSUPPORTED_ANSWER = "unsupported_answer"
    WEAK_RETRIEVAL = "weak_retrieval"
    MISSING_CITATION = "missing_citation"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    OVERCONFIDENT_ANSWER = "overconfident_answer"
    INCOMPLETE_ANSWER = "incomplete_answer"
    NONE = "none"


class PromptStrategy(str, Enum):
    BASELINE = "baseline"
    GROUNDED = "grounded"


class EmbeddingBackend(str, Enum):
    TFIDF = "tfidf"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


# ── Corpus schemas ────────────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """A raw source document in the corpus."""

    doc_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title")
    source_type: str = Field(description="Type of source (e.g., 'policy', 'faq', 'guide')")
    category: str = Field(description="Domain category (e.g., 'billing', 'onboarding')")
    full_text: str = Field(description="Full document text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


class DocumentChunk(BaseModel):
    """A processed chunk derived from a SourceDocument."""

    chunk_id: str = Field(description="Unique chunk identifier (e.g., 'doc001_chunk_02')")
    doc_id: str = Field(description="Parent document ID")
    chunk_text: str = Field(description="Text content of this chunk")
    chunk_order: int = Field(description="Position of chunk within its document (0-indexed)")
    word_count: int = Field(description="Number of words in this chunk")
    token_count_estimate: int = Field(description="Estimated token count (words * 1.33)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Inherited doc metadata")

    @model_validator(mode="after")
    def compute_counts(self) -> "DocumentChunk":
        if self.word_count == 0:
            object.__setattr__(self, "word_count", len(self.chunk_text.split()))
        if self.token_count_estimate == 0:
            object.__setattr__(
                self, "token_count_estimate", int(self.word_count * 1.33)
            )
        return self


# ── Eval set schemas ──────────────────────────────────────────────────────────


class EvalExample(BaseModel):
    """A single evaluation example for the RAG pipeline."""

    example_id: str = Field(description="Unique example identifier")
    question: str = Field(description="The natural-language question")
    reference_answer: str = Field(description="Gold-standard reference answer")
    expected_doc_ids: list[str] = Field(
        default_factory=list, description="Doc IDs that contain the answer"
    )
    expected_key_points: list[str] = Field(
        default_factory=list, description="Key facts that a good answer should mention"
    )
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM)
    category: QuestionCategory = Field(default=QuestionCategory.FACTUAL)
    is_answerable: bool = Field(
        default=True, description="False if the question cannot be answered from corpus"
    )
    notes: str = Field(default="", description="Curator notes")


# ── Pipeline output schemas ───────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its similarity score."""

    chunk_id: str
    doc_id: str
    chunk_text: str
    score: float = Field(ge=0.0, le=1.0, description="Normalized similarity score [0,1]")
    rank: int = Field(ge=1, description="Rank in retrieval result (1 = best)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    """Output from the generation layer."""

    answer_text: str = Field(description="The generated answer")
    cited_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunk IDs cited in the answer"
    )
    cited_doc_ids: list[str] = Field(
        default_factory=list, description="Doc IDs cited in the answer"
    )
    abstained: bool = Field(
        default=False, description="True if the generator chose to abstain"
    )
    prompt_strategy: PromptStrategy = Field(default=PromptStrategy.BASELINE)
    word_count: int = Field(default=0)
    token_count_estimate: int = Field(default=0)

    @model_validator(mode="after")
    def compute_counts(self) -> "GeneratedAnswer":
        if self.word_count == 0:
            object.__setattr__(self, "word_count", len(self.answer_text.split()))
        if self.token_count_estimate == 0:
            object.__setattr__(
                self, "token_count_estimate", int(self.word_count * 1.33)
            )
        return self


# ── Evaluation metric schemas ─────────────────────────────────────────────────


class RetrievalMetrics(BaseModel):
    """Metrics evaluating retrieval quality."""

    hit_at_k: float = Field(ge=0.0, le=1.0, description="Whether any expected doc was retrieved")
    reciprocal_rank: float = Field(ge=0.0, le=1.0, description="1/rank of first relevant doc")
    context_precision: float = Field(ge=0.0, le=1.0, description="Fraction of retrieved chunks that are relevant")
    context_recall: float = Field(ge=0.0, le=1.0, description="Fraction of expected docs that were retrieved")
    retrieved_relevant_count: int = Field(ge=0, description="Count of relevant docs in retrieved set")
    retrieved_total_count: int = Field(ge=0, description="Total retrieved chunks")


class AnswerMetrics(BaseModel):
    """Metrics evaluating answer quality."""

    answer_relevance_score: float = Field(ge=0.0, le=1.0)
    groundedness_score: float = Field(ge=0.0, le=1.0)
    citation_coverage_score: float = Field(ge=0.0, le=1.0)
    exact_match_proxy: float = Field(ge=0.0, le=1.0)
    faithfulness_proxy: float = Field(ge=0.0, le=1.0)
    hallucination_risk_score: float = Field(ge=0.0, le=1.0)
    abstention_quality_score: float = Field(ge=0.0, le=1.0, description="Quality of abstention (1.0 if correct)")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted composite score")


class RunMetrics(BaseModel):
    """Composite metrics for a single pipeline run."""

    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    latency_ms: float = Field(ge=0.0)
    prompt_tokens_estimate: int = Field(ge=0)
    output_tokens_estimate: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ── Trace / run record schema ─────────────────────────────────────────────────


class RunRecord(BaseModel):
    """Complete trace record for one RAG pipeline execution."""

    run_id: str = Field(description="Unique run identifier (UUID)")
    experiment_id: str = Field(default="default", description="Experiment group this run belongs to")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Inputs
    query: str
    example_id: Optional[str] = Field(default=None, description="Eval example ID if applicable")
    prompt_strategy: PromptStrategy = Field(default=PromptStrategy.BASELINE)
    top_k: int = Field(default=5)
    embedding_backend: str = Field(default="tfidf")

    # Pipeline outputs
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    prompt_text: str = Field(default="", description="Full prompt sent to generator")
    generated_answer: Optional[GeneratedAnswer] = None

    # Metrics
    metrics: Optional[RunMetrics] = None

    # Provenance
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of relevant config at run time"
    )
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def overall_score(self) -> Optional[float]:
        if self.metrics:
            return self.metrics.answer.overall_score
        return None

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for DataFrame rows."""
        m = self.metrics
        ans = m.answer if m else None
        ret = m.retrieval if m else None
        gen = self.generated_answer
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "example_id": self.example_id,
            "query": self.query,
            "prompt_strategy": self.prompt_strategy.value,
            "top_k": self.top_k,
            "embedding_backend": self.embedding_backend,
            "answer": gen.answer_text if gen else "",
            "abstained": gen.abstained if gen else False,
            "answer_word_count": gen.word_count if gen else 0,
            "latency_ms": m.latency_ms if m else None,
            "estimated_cost_usd": m.estimated_cost_usd if m else None,
            "overall_score": ans.overall_score if ans else None,
            "answer_relevance": ans.answer_relevance_score if ans else None,
            "groundedness": ans.groundedness_score if ans else None,
            "citation_coverage": ans.citation_coverage_score if ans else None,
            "faithfulness_proxy": ans.faithfulness_proxy if ans else None,
            "hallucination_risk": ans.hallucination_risk_score if ans else None,
            "hit_at_k": ret.hit_at_k if ret else None,
            "reciprocal_rank": ret.reciprocal_rank if ret else None,
            "context_precision": ret.context_precision if ret else None,
            "context_recall": ret.context_recall if ret else None,
            "failure_modes": "|".join(fm.value for fm in m.failure_modes) if m else "",
            "has_errors": len(self.errors) > 0,
            "has_warnings": len(self.warnings) > 0,
        }
