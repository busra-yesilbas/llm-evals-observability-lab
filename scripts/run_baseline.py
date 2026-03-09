"""
run_baseline.py — Run the default RAG pipeline over the eval set.

Executes the baseline configuration (TF-IDF retriever + local generator +
baseline prompt) over all eval examples and stores traces and metrics.

Run from project root:
    python scripts/run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from llm_evals_lab.config import load_config
from llm_evals_lab.data.evalset import EvalSetLoader
from llm_evals_lab.data.loader import CorpusLoader
from llm_evals_lab.generation.rag_pipeline import RAGPipeline
from llm_evals_lab.utils import setup_logging


def main() -> None:
    setup_logging()
    cfg = load_config()

    print("=" * 60)
    print("LLM Evals Lab — Baseline Evaluation Run")
    print("=" * 60)

    # Load corpus chunks
    loader = CorpusLoader(cfg.raw_dir(), cfg.processed_dir())
    chunks = loader.load_chunks()
    if not chunks:
        print("ERROR: No chunks found. Run 'python scripts/prepare_data.py' first.")
        sys.exit(1)
    print(f"✓ Loaded {len(chunks)} corpus chunks")

    # Load eval set
    eval_loader = EvalSetLoader(cfg.eval_dir())
    examples = eval_loader.load()
    if not examples:
        print("ERROR: Eval set not found. Run 'python scripts/build_eval_set.py' first.")
        sys.exit(1)
    print(f"✓ Loaded {len(examples)} eval examples")

    # Build pipeline
    pipeline = RAGPipeline.from_config(cfg, chunks=chunks, experiment_id="baseline")
    print(f"✓ Pipeline ready (retriever: {pipeline.retriever.embedding_backend})")
    print(f"  Prompt strategy: {pipeline.prompt_strategy.value}")
    print(f"  Top-K: {pipeline.top_k}")

    print(f"\nRunning {len(examples)} eval examples...")
    print("-" * 60)

    records = pipeline.run_batch(examples, save_traces=True)

    # Print per-example results
    for rec in records:
        score = rec.overall_score
        score_str = f"{score:.3f}" if score is not None else "N/A"
        ans_text = rec.generated_answer.answer_text[:60] + "..." if rec.generated_answer else "NO ANSWER"
        abstained = rec.generated_answer.abstained if rec.generated_answer else False
        flag = "ABSTAIN" if abstained else "     "
        print(f"  {rec.example_id:10s} | score={score_str} | {flag} | {ans_text}")

    # Aggregate metrics
    summary_df = pipeline.run_store.load_summary()
    baseline_df = summary_df[summary_df["experiment_id"] == "baseline"]

    if not baseline_df.empty:
        print(f"\n{'=' * 60}")
        print("Aggregate Results (baseline)")
        print(f"{'=' * 60}")
        metric_cols = [
            "overall_score", "answer_relevance", "groundedness",
            "citation_coverage", "faithfulness_proxy", "hit_at_k",
            "context_recall", "latency_ms",
        ]
        for col in metric_cols:
            if col in baseline_df.columns:
                vals = baseline_df[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:30s}: mean={vals.mean():.3f}  std={vals.std():.3f}  min={vals.min():.3f}  max={vals.max():.3f}")

        # Failure mode summary
        if "failure_modes" in baseline_df.columns:
            from collections import Counter
            all_modes: list[str] = []
            for fm in baseline_df["failure_modes"].dropna():
                all_modes.extend([m.strip() for m in str(fm).split("|") if m.strip()])
            mode_counts = Counter(all_modes)
            print(f"\nFailure mode counts:")
            for mode, cnt in mode_counts.most_common():
                print(f"  {mode}: {cnt}")

    # Save table
    tables_dir = cfg.tables_dir()
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "baseline_results.csv"
    if not baseline_df.empty:
        baseline_df.to_csv(out_path, index=False)
        print(f"\n✓ Results saved → {out_path}")

    print(f"\n✓ Baseline run complete. {len(records)} traces stored in {cfg.runs_dir()}")


if __name__ == "__main__":
    main()
