"""
analyze_results.py — Aggregate metrics, failure analysis, and figure generation.

Reads stored run traces, computes summary statistics, identifies failure patterns,
and generates publication-quality figures in results/figures/.

Run from project root:
    python scripts/analyze_results.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from llm_evals_lab.config import load_config
from llm_evals_lab.experiments.compare import ExperimentComparison, compare_experiments
from llm_evals_lab.observability.run_store import RunStore
from llm_evals_lab.utils import setup_logging
from llm_evals_lab.visualization.plots import save_all_figures


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main() -> None:
    setup_logging()
    cfg = load_config()

    print_section("LLM Evals Lab — Results Analysis")

    run_store = RunStore(cfg.runs_dir(), cfg.tables_dir())
    df = run_store.to_dataframe()

    if df.empty:
        print("No run data found. Run the pipeline scripts first:")
        print("  python scripts/run_baseline.py")
        print("  python scripts/run_experiments.py")
        sys.exit(0)

    print(f"Loaded {len(df)} total run records")
    if "experiment_id" in df.columns:
        print(f"Experiments: {sorted(df['experiment_id'].unique().tolist())}")

    # ── 1. Overall metric summary ─────────────────────────────────────────────
    print_section("1. Metric Summary (all experiments)")

    metric_cols = [
        "overall_score", "answer_relevance", "groundedness",
        "citation_coverage", "faithfulness_proxy", "hit_at_k",
        "context_recall", "latency_ms",
    ]
    available_metrics = [c for c in metric_cols if c in df.columns]
    if available_metrics:
        summary = df[available_metrics].describe().T.round(4)
        print(summary[["mean", "std", "min", "max"]].to_string())

    # ── 2. Leaderboard ────────────────────────────────────────────────────────
    print_section("2. Experiment Leaderboard")

    if "experiment_id" in df.columns:
        comp = ExperimentComparison(df)
        leaderboard = comp.leaderboard()
        print(leaderboard.to_string(index=False))
        leaderboard.to_csv(cfg.tables_dir() / "leaderboard.csv", index=False)

    # ── 3. Failure mode analysis ──────────────────────────────────────────────
    print_section("3. Failure Mode Analysis")

    if "failure_modes" in df.columns:
        all_modes: list[str] = []
        for fm in df["failure_modes"].dropna():
            all_modes.extend([m.strip() for m in str(fm).split("|") if m.strip()])

        mode_counts = Counter(all_modes)
        print("Failure mode frequency (all experiments):")
        for mode, cnt in mode_counts.most_common():
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 5)
            print(f"  {mode:30s}: {cnt:3d} ({pct:5.1f}%) {bar}")

        failure_df = pd.DataFrame(mode_counts.items(), columns=["failure_mode", "count"])
        failure_df = failure_df.sort_values("count", ascending=False)
        failure_df["pct"] = (failure_df["count"] / len(df) * 100).round(1)
        failure_df.to_csv(cfg.tables_dir() / "failure_mode_analysis.csv", index=False)
        print(f"\n  Saved → {cfg.tables_dir() / 'failure_mode_analysis.csv'}")

    # ── 4. Per-experiment failure breakdown ───────────────────────────────────
    if "experiment_id" in df.columns and "failure_modes" in df.columns:
        print_section("4. Failure Modes by Experiment")
        comp2 = ExperimentComparison(df)
        fm_pivot = comp2.failure_mode_summary()
        if not fm_pivot.empty:
            print(fm_pivot.to_string(index=False))
            fm_pivot.to_csv(cfg.tables_dir() / "failure_modes_by_experiment.csv", index=False)

    # ── 5. Bottom-K runs (hardest examples) ───────────────────────────────────
    print_section("5. Hardest Examples (lowest overall_score)")

    if "overall_score" in df.columns:
        bottom_k = df.nsmallest(10, "overall_score")[
            ["example_id", "experiment_id", "query", "overall_score",
             "groundedness", "hit_at_k", "failure_modes"]
        ].fillna("N/A")
        print(bottom_k.to_string(index=False))
        bottom_k.to_csv(cfg.tables_dir() / "hardest_examples.csv", index=False)

    # ── 6. Score distribution per category ────────────────────────────────────
    print_section("6. Latency Statistics")

    if "latency_ms" in df.columns:
        lat = df["latency_ms"].dropna()
        print(f"  Mean:    {lat.mean():.1f} ms")
        print(f"  Median:  {lat.median():.1f} ms")
        print(f"  P95:     {lat.quantile(0.95):.1f} ms")
        print(f"  Max:     {lat.max():.1f} ms")

    # ── 7. Generate figures ───────────────────────────────────────────────────
    print_section("7. Generating Figures")

    figures_dir = cfg.figures_dir()
    saved = save_all_figures(df, figures_dir)
    for p in saved:
        print(f"  ✓ {p.name}")

    if not saved:
        print("  (matplotlib unavailable or insufficient data)")

    print(f"\n✓ Analysis complete.")
    print(f"  Tables → {cfg.tables_dir()}")
    print(f"  Figures → {figures_dir}")


if __name__ == "__main__":
    main()
