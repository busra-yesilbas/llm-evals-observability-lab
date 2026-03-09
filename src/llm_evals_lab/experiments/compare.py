"""
Experiment comparison utilities for the LLM Evals Lab.

Compares multiple experiment runs across configurable metric dimensions,
produces ranked leaderboards, statistical summaries, and delta tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Metrics that are meaningful for comparison (higher = better)
SCORE_METRICS = [
    "overall_score",
    "answer_relevance",
    "groundedness",
    "citation_coverage",
    "faithfulness_proxy",
    "hit_at_k",
    "reciprocal_rank",
    "context_precision",
    "context_recall",
]

# Metrics where lower is better
COST_METRICS = [
    "latency_ms",
    "estimated_cost_usd",
    "hallucination_risk",
]


class ExperimentComparison:
    """
    Compare multiple experiment groups from a run summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Flat run summary DataFrame (from RunStore.to_dataframe()).
    experiment_col : str
        Column identifying the experiment group.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        experiment_col: str = "experiment_id",
    ) -> None:
        self.df = df
        self.experiment_col = experiment_col

    def leaderboard(
        self,
        metrics: Optional[list[str]] = None,
        sort_by: str = "overall_score",
    ) -> pd.DataFrame:
        """
        Return a leaderboard DataFrame aggregated by experiment.

        Parameters
        ----------
        metrics : list[str], optional
            Which metric columns to include. Defaults to all score metrics.
        sort_by : str
            Column to sort the leaderboard by (descending).

        Returns
        -------
        pd.DataFrame
            One row per experiment, columns = mean of each metric.
        """
        cols = metrics or SCORE_METRICS
        available = [c for c in cols if c in self.df.columns]

        agg_dict = {c: "mean" for c in available}
        agg_dict["run_id"] = "count"

        grouped = (
            self.df.groupby(self.experiment_col)
            .agg(agg_dict)
            .rename(columns={"run_id": "n_runs"})
            .reset_index()
        )

        # Round for readability
        for col in available:
            if col in grouped.columns:
                grouped[col] = grouped[col].round(4)

        if sort_by in grouped.columns:
            grouped = grouped.sort_values(sort_by, ascending=False)

        return grouped.reset_index(drop=True)

    def delta_table(
        self,
        baseline_experiment: str,
        comparison_experiment: str,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute metric deltas between two experiments (comparison - baseline).

        Positive delta → comparison is better (for score metrics).
        """
        cols = metrics or SCORE_METRICS
        available = [c for c in cols if c in self.df.columns]

        baseline_df = self.df[self.df[self.experiment_col] == baseline_experiment]
        comparison_df = self.df[self.df[self.experiment_col] == comparison_experiment]

        rows = []
        for metric in available:
            if metric not in self.df.columns:
                continue
            b_mean = baseline_df[metric].mean()
            c_mean = comparison_df[metric].mean()
            delta = c_mean - b_mean
            pct = (delta / b_mean * 100) if b_mean != 0 else float("nan")
            rows.append(
                {
                    "metric": metric,
                    f"{baseline_experiment}_mean": round(b_mean, 4),
                    f"{comparison_experiment}_mean": round(c_mean, 4),
                    "delta": round(delta, 4),
                    "delta_pct": round(pct, 2),
                }
            )

        return pd.DataFrame(rows)

    def per_example_comparison(
        self,
        experiment_a: str,
        experiment_b: str,
        metric: str = "overall_score",
    ) -> pd.DataFrame:
        """
        Side-by-side comparison of two experiments for each eval example.

        Returns a DataFrame aligned on example_id.
        """
        a_df = self.df[self.df[self.experiment_col] == experiment_a][
            ["example_id", "query", metric]
        ].rename(columns={metric: f"{metric}_{experiment_a}"})
        b_df = self.df[self.df[self.experiment_col] == experiment_b][
            ["example_id", metric]
        ].rename(columns={metric: f"{metric}_{experiment_b}"})

        merged = a_df.merge(b_df, on="example_id", how="outer")
        col_a = f"{metric}_{experiment_a}"
        col_b = f"{metric}_{experiment_b}"
        if col_a in merged.columns and col_b in merged.columns:
            merged["delta"] = (merged[col_b] - merged[col_a]).round(4)
        return merged.sort_values("example_id")

    def failure_mode_summary(self) -> pd.DataFrame:
        """
        Count failure modes per experiment.

        Returns a cross-tab of experiment × failure mode counts.
        """
        if "failure_modes" not in self.df.columns:
            return pd.DataFrame()

        rows = []
        for _, row in self.df.iterrows():
            modes_str = str(row.get("failure_modes", ""))
            modes = [m.strip() for m in modes_str.split("|") if m.strip()]
            for mode in modes:
                rows.append(
                    {
                        "experiment_id": row.get(self.experiment_col, "unknown"),
                        "failure_mode": mode,
                    }
                )

        if not rows:
            return pd.DataFrame()

        mode_df = pd.DataFrame(rows)
        pivot = (
            mode_df.groupby(["experiment_id", "failure_mode"])
            .size()
            .reset_index(name="count")
            .pivot(index="experiment_id", columns="failure_mode", values="count")
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        return pivot

    def summary_statistics(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """
        Return descriptive statistics (mean, std, min, max, p25, p75) for
        all score metrics, optionally filtered by experiment.
        """
        df = self.df if experiment_id is None else self.df[
            self.df[self.experiment_col] == experiment_id
        ]
        cols = [c for c in SCORE_METRICS + COST_METRICS if c in df.columns]
        return df[cols].describe().T.round(4)


def compare_experiments(
    df: pd.DataFrame,
    baseline: str,
    comparisons: list[str],
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run a full comparison suite between baseline and comparison experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Flat run summary DataFrame.
    baseline : str
        Experiment ID to use as reference.
    comparisons : list[str]
        Experiment IDs to compare against the baseline.
    output_dir : Path, optional
        If provided, saves all output tables as CSV files.

    Returns
    -------
    dict mapping table name → DataFrame
    """
    comp = ExperimentComparison(df)
    results: dict[str, pd.DataFrame] = {}

    results["leaderboard"] = comp.leaderboard()
    logger.info("Leaderboard:\n%s", results["leaderboard"].to_string())

    for exp in comparisons:
        key = f"delta_{baseline}_vs_{exp}"
        results[key] = comp.delta_table(baseline, exp)

    results["failure_modes"] = comp.failure_mode_summary()
    results["statistics"] = comp.summary_statistics()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in results.items():
            path = output_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            logger.info("Saved comparison table → %s", path)

    return results
