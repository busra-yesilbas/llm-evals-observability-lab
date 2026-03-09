"""
End-to-end pipeline integration tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evals_lab.data.chunking import Chunker
from llm_evals_lab.data.loader import CorpusLoader
from llm_evals_lab.evaluation.evaluator import Evaluator
from llm_evals_lab.generation.generator import LocalGenerator
from llm_evals_lab.generation.prompts import PromptStrategy, build_prompt
from llm_evals_lab.generation.rag_pipeline import RAGPipeline
from llm_evals_lab.observability.run_store import RunStore
from llm_evals_lab.retrieval.retriever import Retriever
from llm_evals_lab.schemas import FailureMode, RunRecord


class TestEndToEndPipeline:
    @pytest.fixture
    def pipeline(self, sample_chunks, tmp_dir):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=3)
        generator = LocalGenerator(abstain_threshold=0.02)
        evaluator = Evaluator()
        run_store = RunStore(tmp_dir / "runs", tmp_dir / "tables")
        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            evaluator=evaluator,
            run_store=run_store,
            experiment_id="test_exp",
            top_k=3,
            prompt_strategy=PromptStrategy.BASELINE,
        )

    def test_single_run_returns_record(self, pipeline):
        record = pipeline.run("What is the refund policy?", save_trace=False)
        assert isinstance(record, RunRecord)
        assert record.run_id
        assert record.query == "What is the refund policy?"

    def test_run_populates_retrieved_chunks(self, pipeline):
        record = pipeline.run("refund billing", save_trace=False)
        assert len(record.retrieved_chunks) > 0

    def test_run_generates_answer(self, pipeline):
        record = pipeline.run("How many seats in Business plan?", save_trace=False)
        assert record.generated_answer is not None
        assert record.generated_answer.answer_text.strip() != ""

    def test_run_with_eval_example_computes_metrics(self, pipeline, sample_eval_examples):
        example = sample_eval_examples[0]
        record = pipeline.run(example.question, eval_example=example, save_trace=False)
        assert record.metrics is not None
        assert record.metrics.answer.overall_score >= 0.0

    def test_run_saves_trace(self, pipeline, tmp_dir):
        record = pipeline.run("test query", save_trace=True)
        json_path = tmp_dir / "runs" / f"{record.run_id}.json"
        assert json_path.exists()

    def test_batch_run(self, pipeline, sample_eval_examples):
        records = pipeline.run_batch(sample_eval_examples, save_traces=False)
        assert len(records) == len(sample_eval_examples)
        assert all(isinstance(r, RunRecord) for r in records)

    def test_unanswerable_question_may_abstain(self, pipeline, sample_eval_examples):
        unanswerable = sample_eval_examples[2]
        record = pipeline.run(unanswerable.question, save_trace=False)
        # With low context overlap, system should abstain
        assert record.generated_answer is not None
        # Not necessarily abstained (depends on overlap), but should not crash

    def test_grounded_strategy_run(self, pipeline, sample_eval_examples):
        record = pipeline.run(
            sample_eval_examples[0].question,
            eval_example=sample_eval_examples[0],
            prompt_strategy=PromptStrategy.GROUNDED,
            save_trace=False,
        )
        assert record.prompt_strategy == PromptStrategy.GROUNDED
        assert record.generated_answer is not None

    def test_run_record_has_prompt_text(self, pipeline):
        record = pipeline.run("billing refund question", save_trace=False)
        assert len(record.prompt_text) > 0

    def test_run_metrics_failure_modes_populated(self, pipeline, sample_eval_examples):
        record = pipeline.run(
            sample_eval_examples[0].question,
            eval_example=sample_eval_examples[0],
            save_trace=False,
        )
        if record.metrics:
            assert len(record.metrics.failure_modes) > 0


class TestRunStore:
    def test_save_and_load_run(self, sample_chunks, tmp_dir, sample_eval_examples):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=3)
        generator = LocalGenerator()
        evaluator = Evaluator()
        run_store = RunStore(tmp_dir / "runs", tmp_dir / "tables")

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            evaluator=evaluator,
            run_store=run_store,
        )
        record = pipeline.run(
            sample_eval_examples[0].question,
            eval_example=sample_eval_examples[0],
            save_trace=True,
        )

        loaded = run_store.load(record.run_id)
        assert loaded is not None
        assert loaded.run_id == record.run_id
        assert loaded.query == record.query

    def test_load_summary_dataframe(self, sample_chunks, tmp_dir, sample_eval_examples):
        retriever = Retriever.build(sample_chunks, embedding_backend="tfidf", top_k=3)
        run_store = RunStore(tmp_dir / "runs", tmp_dir / "tables")
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=LocalGenerator(),
            evaluator=Evaluator(),
            run_store=run_store,
        )
        for ex in sample_eval_examples[:2]:
            pipeline.run(ex.question, eval_example=ex, save_trace=True)

        df = run_store.to_dataframe()
        assert len(df) >= 2
        assert "run_id" in df.columns
        assert "overall_score" in df.columns

    def test_run_count(self, sample_chunks, tmp_dir, sample_eval_examples):
        run_store = RunStore(tmp_dir / "runs", tmp_dir / "tables")
        pipeline = RAGPipeline(
            retriever=Retriever.build(sample_chunks, embedding_backend="tfidf"),
            generator=LocalGenerator(),
            evaluator=Evaluator(),
            run_store=run_store,
        )
        initial_count = run_store.run_count
        pipeline.run("test", save_trace=True)
        assert run_store.run_count == initial_count + 1


class TestPromptBuilding:
    def test_baseline_prompt_contains_question(self, sample_retrieved_chunks):
        prompt = build_prompt(
            "What is the refund policy?",
            sample_retrieved_chunks,
            strategy=PromptStrategy.BASELINE,
        )
        assert "What is the refund policy?" in prompt
        assert "CONTEXT" in prompt.upper() or "context" in prompt

    def test_grounded_prompt_contains_chunk_ids(self, sample_retrieved_chunks):
        prompt = build_prompt(
            "test question",
            sample_retrieved_chunks,
            strategy=PromptStrategy.GROUNDED,
        )
        assert sample_retrieved_chunks[0].chunk_id in prompt

    def test_empty_chunks_produces_no_context_message(self):
        prompt = build_prompt("test question", [], strategy=PromptStrategy.BASELINE)
        assert "No relevant context" in prompt
