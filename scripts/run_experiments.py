"""
run_experiments.py — Run a grid of experiments and compare configurations.

Experiments vary:
  1. top_k: 3 vs 5
  2. prompt_strategy: baseline vs grounded
  3. abstain_threshold variants (via generator config)

Each experiment runs over the full eval set and saves traces to results/runs/.

Run from project root:
    python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_evals_lab.config import load_config
from llm_evals_lab.data.evalset import EvalSetLoader
from llm_evals_lab.data.loader import CorpusLoader
from llm_evals_lab.evaluation.evaluator import Evaluator
from llm_evals_lab.experiments.compare import compare_experiments
from llm_evals_lab.generation.generator import LocalGenerator
from llm_evals_lab.generation.prompts import PromptStrategy
from llm_evals_lab.generation.rag_pipeline import RAGPipeline
from llm_evals_lab.observability.run_store import RunStore
from llm_evals_lab.retrieval.retriever import Retriever
from llm_evals_lab.utils import setup_logging


EXPERIMENTS = [
    {
        "id": "baseline_k5",
        "top_k": 5,
        "prompt_strategy": PromptStrategy.BASELINE,
        "abstain_threshold": 0.03,
        "description": "Baseline: top-5, concise QA prompt",
    },
    {
        "id": "baseline_k3",
        "top_k": 3,
        "prompt_strategy": PromptStrategy.BASELINE,
        "abstain_threshold": 0.03,
        "description": "Top-3 retrieval, baseline prompt",
    },
    {
        "id": "grounded_k5",
        "top_k": 5,
        "prompt_strategy": PromptStrategy.GROUNDED,
        "abstain_threshold": 0.03,
        "description": "Citation-first grounded prompt, top-5",
    },
    {
        "id": "grounded_k3",
        "top_k": 3,
        "prompt_strategy": PromptStrategy.GROUNDED,
        "abstain_threshold": 0.03,
        "description": "Citation-first grounded prompt, top-3",
    },
    {
        "id": "high_abstain_k5",
        "top_k": 5,
        "prompt_strategy": PromptStrategy.BASELINE,
        "abstain_threshold": 0.10,
        "description": "Higher abstention threshold (conservative)",
    },
]


def run_experiment(
    exp: dict,
    chunks,
    examples,
    cfg,
    run_store: RunStore,
) -> None:
    """Run a single experiment configuration over all eval examples."""
    exp_id = exp["id"]
    print(f"\n[{exp_id}] {exp['description']}")
    print(f"  top_k={exp['top_k']}, strategy={exp['prompt_strategy'].value}, "
          f"abstain_threshold={exp['abstain_threshold']}")

    # Build retriever (reuse index if already built)
    ret_cfg = cfg._raw.copy()
    idx_path = cfg.processed_dir() / "retrieval_index.pkl"
    retriever = Retriever.from_config(chunks, ret_cfg, index_path=idx_path)

    generator = LocalGenerator(abstain_threshold=exp["abstain_threshold"])
    evaluator = Evaluator(cfg._raw.get("eval", {}))

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        run_store=run_store,
        experiment_id=exp_id,
        top_k=exp["top_k"],
        prompt_strategy=exp["prompt_strategy"],
    )

    records = pipeline.run_batch(examples, save_traces=True)

    # Quick summary
    scores = [r.overall_score for r in records if r.overall_score is not None]
    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"  → {len(records)} runs | mean overall_score = {mean_score:.3f}")
    else:
        print(f"  → {len(records)} runs | no scores computed")


def main() -> None:
    setup_logging()
    cfg = load_config()

    print("=" * 60)
    print("LLM Evals Lab — Multi-Experiment Runner")
    print("=" * 60)

    # Load data
    loader = CorpusLoader(cfg.raw_dir(), cfg.processed_dir())
    chunks = loader.load_chunks()
    if not chunks:
        print("ERROR: No chunks found. Run 'python scripts/prepare_data.py' first.")
        sys.exit(1)

    eval_loader = EvalSetLoader(cfg.eval_dir())
    examples = eval_loader.load()
    if not examples:
        print("ERROR: Eval set not found. Run 'python scripts/build_eval_set.py' first.")
        sys.exit(1)

    run_store = RunStore(cfg.runs_dir(), cfg.tables_dir())
    print(f"✓ Loaded {len(chunks)} chunks, {len(examples)} eval examples")
    print(f"Running {len(EXPERIMENTS)} experiments × {len(examples)} examples = "
          f"{len(EXPERIMENTS) * len(examples)} total runs\n")

    # Run all experiments
    for exp in EXPERIMENTS:
        run_experiment(exp, chunks, examples, cfg, run_store)

    # Compare experiments
    print("\n" + "=" * 60)
    print("Experiment Comparison")
    print("=" * 60)

    df = run_store.to_dataframe()
    if df.empty:
        print("No data to compare.")
        return

    exp_ids = [e["id"] for e in EXPERIMENTS]
    comparison_ids = [e for e in exp_ids if e != "baseline_k5"]

    results = compare_experiments(
        df=df[df["experiment_id"].isin(exp_ids)],
        baseline="baseline_k5",
        comparisons=comparison_ids,
        output_dir=cfg.tables_dir(),
    )

    print("\nLeaderboard:")
    print(results["leaderboard"].to_string(index=False))

    if "delta_baseline_k5_vs_grounded_k5" in results:
        print("\nDelta: grounded_k5 vs baseline_k5:")
        delta = results["delta_baseline_k5_vs_grounded_k5"]
        print(delta[["metric", "delta", "delta_pct"]].to_string(index=False))

    print(f"\n✓ Experiment results saved to {cfg.tables_dir()}")


if __name__ == "__main__":
    main()
