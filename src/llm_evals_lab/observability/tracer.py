"""
Run tracer for the LLM Evals Lab.

The Tracer accumulates state throughout a pipeline execution and produces
a fully-populated RunRecord at the end. It also tracks per-step timing.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from llm_evals_lab.schemas import (
    GeneratedAnswer,
    PromptStrategy,
    RetrievedChunk,
    RunMetrics,
    RunRecord,
)


class Tracer:
    """
    Mutable accumulator that builds a RunRecord incrementally.

    A Tracer is created at the start of each pipeline run and populated
    as each step completes. Call ``build_record()`` at the end to get
    the immutable RunRecord.
    """

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        query: str,
        example_id: Optional[str] = None,
        prompt_strategy: PromptStrategy = PromptStrategy.BASELINE,
        top_k: int = 5,
        embedding_backend: str = "tfidf",
    ) -> None:
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.query = query
        self.example_id = example_id
        self.prompt_strategy = prompt_strategy
        self.top_k = top_k
        self.embedding_backend = embedding_backend
        self.timestamp = datetime.utcnow()

        # Filled in progressively
        self._retrieved_chunks: list[RetrievedChunk] = []
        self._retrieved_doc_ids: list[str] = []
        self._prompt_text: str = ""
        self._generated_answer: Optional[GeneratedAnswer] = None
        self._metrics: Optional[RunMetrics] = None
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._latency_ms: float = 0.0

        # Step timing
        self._step_starts: dict[str, float] = {}
        self._step_durations: dict[str, float] = {}

    # ── Step timing ───────────────────────────────────────────────────────────

    def start_step(self, step: str) -> None:
        self._step_starts[step] = time.perf_counter()

    def end_step(self, step: str) -> None:
        if step in self._step_starts:
            self._step_durations[step] = (
                time.perf_counter() - self._step_starts[step]
            ) * 1000.0

    # ── State setters ─────────────────────────────────────────────────────────

    def set_retrieved(self, chunks: list[RetrievedChunk]) -> None:
        self._retrieved_chunks = chunks
        self._retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))

    def set_prompt(self, prompt_text: str) -> None:
        self._prompt_text = prompt_text

    def set_answer(self, answer: GeneratedAnswer) -> None:
        self._generated_answer = answer

    def set_metrics(self, metrics: RunMetrics) -> None:
        self._metrics = metrics

    def set_latency(self, latency_ms: float) -> None:
        self._latency_ms = latency_ms

    def add_error(self, error: str) -> None:
        self._errors.append(error)

    def add_warning(self, warning: str) -> None:
        self._warnings.append(warning)

    # ── Record assembly ───────────────────────────────────────────────────────

    def build_record(self) -> RunRecord:
        """
        Assemble the current tracer state into a RunRecord.

        This can be called multiple times (e.g., before and after evaluation).
        """
        # Merge any metric-level warnings
        warnings = list(self._warnings)
        if self._metrics and self._metrics.warnings:
            warnings.extend(self._metrics.warnings)

        # Include step timings in config snapshot
        config_snapshot: dict = {
            "experiment_id": self.experiment_id,
            "top_k": self.top_k,
            "prompt_strategy": self.prompt_strategy.value,
            "embedding_backend": self.embedding_backend,
            "step_durations_ms": {
                k: round(v, 2) for k, v in self._step_durations.items()
            },
        }

        # If we have latency from the outer timer, add it to metrics
        metrics = self._metrics
        if metrics and self._latency_ms > 0:
            from llm_evals_lab.schemas import RunMetrics
            metrics = RunMetrics(
                retrieval=metrics.retrieval,
                answer=metrics.answer,
                latency_ms=self._latency_ms,
                prompt_tokens_estimate=metrics.prompt_tokens_estimate,
                output_tokens_estimate=metrics.output_tokens_estimate,
                estimated_cost_usd=metrics.estimated_cost_usd,
                failure_modes=metrics.failure_modes,
                warnings=metrics.warnings,
            )

        return RunRecord(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            timestamp=self.timestamp,
            query=self.query,
            example_id=self.example_id,
            prompt_strategy=self.prompt_strategy,
            top_k=self.top_k,
            embedding_backend=self.embedding_backend,
            retrieved_chunks=self._retrieved_chunks,
            retrieved_doc_ids=self._retrieved_doc_ids,
            prompt_text=self._prompt_text,
            generated_answer=self._generated_answer,
            metrics=metrics,
            config_snapshot=config_snapshot,
            errors=self._errors,
            warnings=warnings,
        )
