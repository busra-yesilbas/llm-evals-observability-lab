"""
End-to-end RAG pipeline for the LLM Evals Lab.

Orchestrates: retrieval → prompt building → generation → tracing → scoring.
Every run produces a RunRecord that is optionally persisted to disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from llm_evals_lab.config import LabConfig
from llm_evals_lab.data.loader import CorpusLoader
from llm_evals_lab.evaluation.evaluator import Evaluator
from llm_evals_lab.generation.generator import BaseGenerator, get_generator
from llm_evals_lab.generation.prompts import PromptStrategy, build_prompt
from llm_evals_lab.observability.run_store import RunStore
from llm_evals_lab.observability.tracer import Tracer
from llm_evals_lab.retrieval.retriever import Retriever
from llm_evals_lab.schemas import (
    DocumentChunk,
    EvalExample,
    RunRecord,
)
from llm_evals_lab.utils import new_run_id, setup_logging, timer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Parameters
    ----------
    retriever : Retriever
        Fitted retrieval engine.
    generator : BaseGenerator
        Answer generator (local or API-backed).
    evaluator : Evaluator
        Metric computation engine.
    run_store : RunStore
        Persistence layer for run traces.
    experiment_id : str
        Label for this experiment group.
    top_k : int
        Default number of chunks to retrieve.
    prompt_strategy : PromptStrategy
        Default prompt template.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: BaseGenerator,
        evaluator: Evaluator,
        run_store: RunStore,
        experiment_id: str = "default",
        top_k: int = 5,
        prompt_strategy: PromptStrategy = PromptStrategy.BASELINE,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
        self.run_store = run_store
        self.experiment_id = experiment_id
        self.top_k = top_k
        self.prompt_strategy = prompt_strategy

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: LabConfig,
        chunks: Optional[list[DocumentChunk]] = None,
        experiment_id: str = "default",
    ) -> "RAGPipeline":
        """Build a fully-configured RAGPipeline from a LabConfig object."""
        setup_logging(cfg._raw.get("logging", {}).get("level", "INFO"))

        if chunks is None:
            loader = CorpusLoader(cfg.raw_dir(), cfg.processed_dir())
            chunks = loader.load_chunks()
            if not chunks:
                raise RuntimeError(
                    "No chunks found. Run: python scripts/prepare_data.py first."
                )

        ret_cfg = cfg.retrieval
        index_path: Optional[Path] = None
        if ret_cfg.get("index", {}).get("persist", True):
            idx_dir = cfg.processed_dir()
            idx_file = ret_cfg.get("index", {}).get("index_filename", "retrieval_index.pkl")
            index_path = idx_dir / idx_file

        retriever = Retriever.from_config(chunks, cfg._raw, index_path=index_path)

        gen_cfg = cfg.generation
        generator = get_generator(
            backend=gen_cfg.get("backend", "local"),
            model=gen_cfg.get("model"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        evaluator = Evaluator(cfg._raw.get("eval", {}))
        run_store = RunStore(cfg.runs_dir())

        strategy_str = cfg._raw.get("generation", {}).get("default_prompt_strategy", "baseline")
        strategy = PromptStrategy(strategy_str) if strategy_str in PromptStrategy._value2member_map_ else PromptStrategy.BASELINE

        return cls(
            retriever=retriever,
            generator=generator,
            evaluator=evaluator,
            run_store=run_store,
            experiment_id=experiment_id,
            top_k=ret_cfg.get("top_k", 5),
            prompt_strategy=strategy,
        )

    # ── Main run method ───────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        eval_example: Optional[EvalExample] = None,
        top_k: Optional[int] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        save_trace: bool = True,
    ) -> RunRecord:
        """
        Execute one full RAG pipeline pass.

        Parameters
        ----------
        query : str
            The user question.
        eval_example : EvalExample, optional
            Provide to enable metric computation.
        top_k : int, optional
            Override default top_k.
        prompt_strategy : PromptStrategy, optional
            Override default prompt strategy.
        save_trace : bool
            Whether to persist the RunRecord to disk.

        Returns
        -------
        RunRecord
            Complete trace including retrieved chunks, answer, and metrics.
        """
        k = top_k or self.top_k
        strategy = prompt_strategy or self.prompt_strategy
        run_id = new_run_id()

        tracer = Tracer(
            run_id=run_id,
            experiment_id=self.experiment_id,
            query=query,
            example_id=eval_example.example_id if eval_example else None,
            prompt_strategy=strategy,
            top_k=k,
            embedding_backend=self.retriever.embedding_backend,
        )

        with timer() as t:
            try:
                # ── Step 1: Retrieve ──────────────────────────────────────────
                tracer.start_step("retrieval")
                retrieved_chunks = self.retriever.retrieve(query, top_k=k)
                tracer.end_step("retrieval")
                tracer.set_retrieved(retrieved_chunks)
                logger.debug("Retrieved %d chunks for query: %r", len(retrieved_chunks), query)

                # ── Step 2: Build prompt ──────────────────────────────────────
                tracer.start_step("prompt_building")
                prompt_text = build_prompt(query, retrieved_chunks, strategy=strategy)
                tracer.end_step("prompt_building")
                tracer.set_prompt(prompt_text)

                # ── Step 3: Generate ──────────────────────────────────────────
                tracer.start_step("generation")
                generated_answer = self.generator.generate(
                    prompt=prompt_text,
                    retrieved_chunks=retrieved_chunks,
                    question=query,
                    strategy=strategy,
                )
                tracer.end_step("generation")
                tracer.set_answer(generated_answer)

            except Exception as exc:
                tracer.add_error(str(exc))
                logger.exception("Pipeline error for run %s: %s", run_id, exc)

        latency_ms = t["ms"]
        tracer.set_latency(latency_ms)

        # ── Step 4: Evaluate (if reference available) ─────────────────────────
        record = tracer.build_record()
        if eval_example and record.generated_answer:
            metrics = self.evaluator.evaluate(record, eval_example)
            tracer.set_metrics(metrics)
            record = tracer.build_record()

        if save_trace:
            self.run_store.save(record)

        return record

    def run_batch(
        self,
        eval_examples: list[EvalExample],
        top_k: Optional[int] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        save_traces: bool = True,
    ) -> list[RunRecord]:
        """
        Run the pipeline over a list of eval examples.

        Returns a list of RunRecords with metrics.
        """
        records: list[RunRecord] = []
        for i, example in enumerate(eval_examples, start=1):
            logger.info("Running example %d/%d: %s", i, len(eval_examples), example.example_id)
            record = self.run(
                query=example.question,
                eval_example=example,
                top_k=top_k,
                prompt_strategy=prompt_strategy,
                save_trace=save_traces,
            )
            records.append(record)
        return records
