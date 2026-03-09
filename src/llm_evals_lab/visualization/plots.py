"""
Visualization functions for the LLM Evals Lab.

Provides both Plotly (interactive, for dashboard) and Matplotlib
(static, for saving to figures/) chart functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Plotly charts (interactive) ───────────────────────────────────────────────

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("plotly not available — interactive charts disabled")


def score_radar_chart(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    metrics: Optional[list[str]] = None,
):
    """
    Radar (spider) chart comparing experiments across metric dimensions.

    Returns a Plotly Figure or None if plotly is unavailable.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    default_metrics = [
        "answer_relevance", "groundedness", "citation_coverage",
        "faithfulness_proxy", "context_precision", "context_recall",
    ]
    cols = metrics or default_metrics
    available = [c for c in cols if c in df.columns]
    if not available:
        return None

    agg = df.groupby(experiment_col)[available].mean().reset_index()

    fig = go.Figure()
    for _, row in agg.iterrows():
        values = [row[c] for c in available]
        values.append(values[0])  # close the polygon
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=available + [available[0]],
                fill="toself",
                name=str(row[experiment_col]),
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Metric Radar Chart by Experiment",
        showlegend=True,
    )
    return fig


def metric_bar_chart(
    df: pd.DataFrame,
    metric: str = "overall_score",
    experiment_col: str = "experiment_id",
    title: Optional[str] = None,
):
    """Bar chart of a single metric aggregated by experiment."""
    if not _PLOTLY_AVAILABLE:
        return None
    if metric not in df.columns:
        return None

    agg = df.groupby(experiment_col)[metric].mean().reset_index()
    agg = agg.sort_values(metric, ascending=False)

    fig = px.bar(
        agg,
        x=experiment_col,
        y=metric,
        color=experiment_col,
        title=title or f"{metric} by Experiment",
        range_y=[0, 1],
        text=agg[metric].round(3),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def score_distribution(
    df: pd.DataFrame,
    metric: str = "overall_score",
    experiment_col: str = "experiment_id",
    title: Optional[str] = None,
):
    """Box plot of score distributions per experiment."""
    if not _PLOTLY_AVAILABLE:
        return None
    if metric not in df.columns:
        return None

    fig = px.box(
        df,
        x=experiment_col,
        y=metric,
        color=experiment_col,
        title=title or f"{metric} Distribution",
        points="all",
    )
    fig.update_layout(yaxis_range=[0, 1], showlegend=False)
    return fig


def latency_histogram(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    title: str = "Latency Distribution (ms)",
):
    """Histogram of latency per experiment."""
    if not _PLOTLY_AVAILABLE:
        return None
    if "latency_ms" not in df.columns:
        return None

    fig = px.histogram(
        df,
        x="latency_ms",
        color=experiment_col,
        barmode="overlay",
        title=title,
        opacity=0.75,
        nbins=30,
    )
    return fig


def failure_mode_chart(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    title: str = "Failure Mode Distribution",
):
    """Stacked bar chart of failure modes by experiment."""
    if not _PLOTLY_AVAILABLE:
        return None
    if "failure_modes" not in df.columns:
        return None

    rows = []
    for _, row in df.iterrows():
        modes_str = str(row.get("failure_modes", ""))
        modes = [m.strip() for m in modes_str.split("|") if m.strip()]
        for mode in modes:
            rows.append({
                experiment_col: row.get(experiment_col, "unknown"),
                "failure_mode": mode,
            })

    if not rows:
        return None

    mode_df = pd.DataFrame(rows)
    counts = mode_df.groupby([experiment_col, "failure_mode"]).size().reset_index(name="count")

    fig = px.bar(
        counts,
        x=experiment_col,
        y="count",
        color="failure_mode",
        barmode="stack",
        title=title,
    )
    return fig


def score_heatmap(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    metrics: Optional[list[str]] = None,
    title: str = "Metric Heatmap by Experiment",
):
    """Heatmap of mean metric scores per experiment."""
    if not _PLOTLY_AVAILABLE:
        return None

    default_metrics = [
        "overall_score", "answer_relevance", "groundedness",
        "citation_coverage", "faithfulness_proxy", "context_precision",
        "context_recall", "hit_at_k",
    ]
    cols = metrics or default_metrics
    available = [c for c in cols if c in df.columns]
    if not available:
        return None

    agg = df.groupby(experiment_col)[available].mean()
    fig = px.imshow(
        agg,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        title=title,
    )
    return fig


def retrieval_score_scatter(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    title: str = "Context Recall vs Answer Relevance",
):
    """Scatter plot of retrieval quality vs answer quality."""
    if not _PLOTLY_AVAILABLE:
        return None

    required = ["context_recall", "answer_relevance"]
    if not all(c in df.columns for c in required):
        return None

    fig = px.scatter(
        df,
        x="context_recall",
        y="answer_relevance",
        color=experiment_col,
        hover_data=["example_id", "overall_score"] if "example_id" in df.columns else None,
        title=title,
        range_x=[0, 1],
        range_y=[0, 1],
        opacity=0.7,
    )
    # Add diagonal reference line
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray"),
    )
    return fig


# ── Matplotlib charts (static, for saving) ───────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def save_metric_bar_chart_mpl(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "overall_score",
    experiment_col: str = "experiment_id",
    title: Optional[str] = None,
) -> Optional[Path]:
    """Save a static bar chart using Matplotlib."""
    if not _MPL_AVAILABLE:
        return None
    if metric not in df.columns:
        return None

    agg = df.groupby(experiment_col)[metric].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(agg.index, agg.values, color="#4C72B0", edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Experiment")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart to %s", output_path)
    return output_path


def save_score_heatmap_mpl(
    df: pd.DataFrame,
    output_path: Path,
    metrics: Optional[list[str]] = None,
    experiment_col: str = "experiment_id",
    title: str = "Metric Heatmap",
) -> Optional[Path]:
    """Save a static heatmap using Matplotlib."""
    if not _MPL_AVAILABLE:
        return None

    default_metrics = [
        "overall_score", "answer_relevance", "groundedness",
        "citation_coverage", "faithfulness_proxy", "context_recall",
    ]
    cols = metrics or default_metrics
    available = [c for c in cols if c in df.columns]
    if not available:
        return None

    agg = df.groupby(experiment_col)[available].mean()
    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.2), max(4, len(agg) * 0.8)))
    cax = ax.imshow(agg.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right")
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index)
    ax.set_title(title)

    for i in range(len(agg)):
        for j in range(len(available)):
            ax.text(j, i, f"{agg.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(cax, ax=ax)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)
    return output_path


def save_all_figures(df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Save a standard set of static figures from a run summary DataFrame."""
    saved: list[Path] = []

    p = save_metric_bar_chart_mpl(df, figures_dir / "overall_score_bar.png", "overall_score")
    if p:
        saved.append(p)

    p = save_score_heatmap_mpl(df, figures_dir / "metric_heatmap.png")
    if p:
        saved.append(p)

    return saved
