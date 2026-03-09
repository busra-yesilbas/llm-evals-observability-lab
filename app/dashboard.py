"""
LLM Evals & Observability Lab — Streamlit Dashboard

Run from project root:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import streamlit as st

from llm_evals_lab.config import load_config
from llm_evals_lab.data.evalset import EvalSetLoader
from llm_evals_lab.data.loader import CorpusLoader
from llm_evals_lab.experiments.compare import ExperimentComparison
from llm_evals_lab.observability.run_store import RunStore

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Evals & Observability Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load config and data (cached) ─────────────────────────────────────────────

@st.cache_resource
def get_config():
    return load_config()

@st.cache_data(ttl=30)
def load_run_data():
    cfg = get_config()
    store = RunStore(cfg.runs_dir(), cfg.tables_dir())
    df = store.to_dataframe()
    return df, store

@st.cache_data(ttl=60)
def load_corpus_data():
    cfg = get_config()
    loader = CorpusLoader(cfg.raw_dir(), cfg.processed_dir())
    docs = loader.load_documents()
    chunks = loader.load_chunks()
    return docs, chunks

@st.cache_data(ttl=60)
def load_eval_data():
    cfg = get_config()
    loader = EvalSetLoader(cfg.eval_dir())
    return loader.load()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/LLM_Evals-Lab-4C72B0?style=for-the-badge", use_column_width=True)
    st.title("🔬 LLM Evals Lab")
    st.caption("Evaluation & Observability Framework")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "📚 Corpus Explorer",
            "🎯 Eval Set Explorer",
            "🏃 Run Explorer",
            "🔍 Trace Viewer",
            "🏆 Metrics Leaderboard",
            "📡 Retrieval Analysis",
            "⚠️ Failure Analysis",
            "🧪 Experiment Comparison",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("v0.1.0 · MIT License")

# ── Load data ─────────────────────────────────────────────────────────────────

cfg = get_config()
df, run_store = load_run_data()
docs, chunks = load_corpus_data()
eval_examples = load_eval_data()

# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Overview")
    st.markdown(
        "**LLM Evals & Observability Lab** — A production-grade framework for evaluating, "
        "tracing, and stress-testing RAG and agentic AI systems."
    )

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Source Documents", len(docs))
    with col2:
        st.metric("Corpus Chunks", len(chunks))
    with col3:
        st.metric("Eval Examples", len(eval_examples))
    with col4:
        st.metric("Total Runs", len(df) if not df.empty else 0)
    with col5:
        if not df.empty and "experiment_id" in df.columns:
            st.metric("Experiments", df["experiment_id"].nunique())
        else:
            st.metric("Experiments", 0)

    st.divider()

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Overall Score by Experiment")
            if "overall_score" in df.columns and "experiment_id" in df.columns:
                agg = df.groupby("experiment_id")["overall_score"].mean().sort_values(ascending=False)
                st.bar_chart(agg)
            else:
                st.info("No scored runs yet. Run: `python scripts/run_baseline.py`")

        with col2:
            st.subheader("Metric Averages (All Experiments)")
            metric_cols = ["overall_score", "answer_relevance", "groundedness",
                           "citation_coverage", "faithfulness_proxy", "hit_at_k", "context_recall"]
            available = [c for c in metric_cols if c in df.columns]
            if available:
                means = df[available].mean().round(3)
                st.dataframe(
                    pd.DataFrame({"metric": means.index, "mean_score": means.values}),
                    use_container_width=True,
                    hide_index=True,
                )

        st.subheader("Recent Runs")
        display_cols = ["run_id", "experiment_id", "timestamp", "example_id",
                        "overall_score", "latency_ms", "failure_modes"]
        show_cols = [c for c in display_cols if c in df.columns]
        if show_cols:
            st.dataframe(
                df[show_cols].tail(20).sort_values("timestamp", ascending=False)
                if "timestamp" in df.columns else df[show_cols].tail(20),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "No run data found yet.\n\n"
            "**Quick start:**\n"
            "```bash\n"
            "python scripts/prepare_data.py\n"
            "python scripts/build_eval_set.py\n"
            "python scripts/run_baseline.py\n"
            "```"
        )

# ── Page: Corpus Explorer ─────────────────────────────────────────────────────

elif page == "📚 Corpus Explorer":
    st.title("📚 Corpus Explorer")

    if not docs:
        st.warning("No corpus loaded. Run `python scripts/prepare_data.py`")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(docs))
        with col2:
            total_words = sum(d.word_count for d in docs)
            st.metric("Total Words", f"{total_words:,}")
        with col3:
            st.metric("Chunks", len(chunks))

        st.subheader("Document Index")
        doc_data = [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "category": d.category,
                "source_type": d.source_type,
                "word_count": d.word_count,
            }
            for d in docs
        ]
        st.dataframe(pd.DataFrame(doc_data), use_container_width=True, hide_index=True)

        st.subheader("Document Viewer")
        selected_doc_id = st.selectbox(
            "Select a document",
            [d.doc_id for d in docs],
            format_func=lambda x: next((f"{d.doc_id} — {d.title}" for d in docs if d.doc_id == x), x),
        )
        selected_doc = next((d for d in docs if d.doc_id == selected_doc_id), None)
        if selected_doc:
            st.markdown(f"**{selected_doc.title}**  |  Category: `{selected_doc.category}`  |  Type: `{selected_doc.source_type}`")
            with st.expander("Full Text", expanded=True):
                st.text(selected_doc.full_text)

            st.subheader(f"Chunks from {selected_doc_id}")
            doc_chunks = [c for c in chunks if c.doc_id == selected_doc_id]
            for chunk in doc_chunks:
                with st.expander(f"[{chunk.chunk_id}] {chunk.word_count} words"):
                    st.text(chunk.chunk_text)

# ── Page: Eval Set Explorer ───────────────────────────────────────────────────

elif page == "🎯 Eval Set Explorer":
    st.title("🎯 Eval Set Explorer")

    if not eval_examples:
        st.warning("Eval set not built. Run `python scripts/build_eval_set.py`")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Examples", len(eval_examples))
        with col2:
            answerable = sum(1 for e in eval_examples if e.is_answerable)
            st.metric("Answerable", answerable)
        with col3:
            st.metric("Unanswerable", len(eval_examples) - answerable)

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            categories = ["All"] + sorted(set(e.category.value for e in eval_examples))
            selected_cat = st.selectbox("Filter by category", categories)
        with col2:
            difficulties = ["All"] + sorted(set(e.difficulty.value for e in eval_examples))
            selected_diff = st.selectbox("Filter by difficulty", difficulties)

        filtered = eval_examples
        if selected_cat != "All":
            filtered = [e for e in filtered if e.category.value == selected_cat]
        if selected_diff != "All":
            filtered = [e for e in filtered if e.difficulty.value == selected_diff]

        eval_data = [
            {
                "example_id": e.example_id,
                "difficulty": e.difficulty.value,
                "category": e.category.value,
                "answerable": e.is_answerable,
                "question": e.question[:80] + "..." if len(e.question) > 80 else e.question,
            }
            for e in filtered
        ]
        st.dataframe(pd.DataFrame(eval_data), use_container_width=True, hide_index=True)

        st.subheader("Example Detail")
        example_ids = [e.example_id for e in filtered]
        if example_ids:
            sel_id = st.selectbox("Select example", example_ids)
            sel_ex = next((e for e in filtered if e.example_id == sel_id), None)
            if sel_ex:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Difficulty", sel_ex.difficulty.value)
                with col2:
                    st.metric("Category", sel_ex.category.value)
                with col3:
                    st.metric("Answerable", "Yes" if sel_ex.is_answerable else "No")

                st.markdown("**Question:**")
                st.info(sel_ex.question)
                st.markdown("**Reference Answer:**")
                st.success(sel_ex.reference_answer)
                if sel_ex.expected_key_points:
                    st.markdown("**Expected Key Points:**")
                    for kp in sel_ex.expected_key_points:
                        st.markdown(f"- {kp}")
                if sel_ex.notes:
                    st.caption(f"Notes: {sel_ex.notes}")

# ── Page: Run Explorer ────────────────────────────────────────────────────────

elif page == "🏃 Run Explorer":
    st.title("🏃 Run Explorer")

    if df.empty:
        st.info("No runs stored yet. Run the pipeline scripts first.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            if "experiment_id" in df.columns:
                experiments = ["All"] + sorted(df["experiment_id"].dropna().unique().tolist())
                sel_exp = st.selectbox("Filter by experiment", experiments)
            else:
                sel_exp = "All"
        with col2:
            if "prompt_strategy" in df.columns:
                strategies = ["All"] + sorted(df["prompt_strategy"].dropna().unique().tolist())
                sel_strat = st.selectbox("Filter by strategy", strategies)
            else:
                sel_strat = "All"

        filtered_df = df.copy()
        if sel_exp != "All" and "experiment_id" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["experiment_id"] == sel_exp]
        if sel_strat != "All" and "prompt_strategy" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["prompt_strategy"] == sel_strat]

        st.metric("Showing runs", len(filtered_df))

        display_cols = [
            "run_id", "experiment_id", "example_id", "timestamp",
            "overall_score", "groundedness", "hit_at_k",
            "latency_ms", "abstained", "failure_modes",
        ]
        show_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[show_cols].sort_values("timestamp", ascending=False)
            if "timestamp" in filtered_df.columns else filtered_df[show_cols],
            use_container_width=True,
            hide_index=True,
        )

# ── Page: Trace Viewer ────────────────────────────────────────────────────────

elif page == "🔍 Trace Viewer":
    st.title("🔍 Trace Viewer")
    st.markdown("Inspect complete run traces including retrieved chunks, prompts, and all metrics.")

    if df.empty:
        st.info("No runs available yet.")
    else:
        run_ids = df["run_id"].tolist() if "run_id" in df.columns else []
        if run_ids:
            sel_run_id = st.selectbox("Select run ID", run_ids[:100])

            record = run_store.load(sel_run_id)
            if record:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Experiment", record.experiment_id)
                with col2:
                    st.metric("Strategy", record.prompt_strategy.value)
                with col3:
                    st.metric("Top-K", record.top_k)
                with col4:
                    if record.metrics:
                        st.metric("Latency (ms)", f"{record.metrics.latency_ms:.1f}")

                st.markdown("**Query:**")
                st.info(record.query)

                if record.generated_answer:
                    ans = record.generated_answer
                    st.markdown("**Generated Answer:**")
                    color = "warning" if ans.abstained else "success"
                    if ans.abstained:
                        st.warning(f"🚫 ABSTAINED: {ans.answer_text}")
                    else:
                        st.success(ans.answer_text)
                    if ans.cited_chunk_ids:
                        st.caption(f"Cited chunks: {', '.join(ans.cited_chunk_ids)}")

                with st.expander("📋 Retrieved Chunks"):
                    for chunk in record.retrieved_chunks:
                        st.markdown(
                            f"**[{chunk.chunk_id}]** rank={chunk.rank} score={chunk.score:.3f}"
                        )
                        st.text(chunk.chunk_text[:400] + "..." if len(chunk.chunk_text) > 400 else chunk.chunk_text)
                        st.divider()

                with st.expander("📝 Full Prompt"):
                    st.code(record.prompt_text, language="text")

                if record.metrics:
                    m = record.metrics
                    st.subheader("Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{m.answer.overall_score:.3f}")
                        st.metric("Answer Relevance", f"{m.answer.answer_relevance_score:.3f}")
                        st.metric("Groundedness", f"{m.answer.groundedness_score:.3f}")
                    with col2:
                        st.metric("Citation Coverage", f"{m.answer.citation_coverage_score:.3f}")
                        st.metric("Faithfulness", f"{m.answer.faithfulness_proxy:.3f}")
                        st.metric("Hallucination Risk", f"{m.answer.hallucination_risk_score:.3f}")
                    with col3:
                        st.metric("Hit@K", f"{m.retrieval.hit_at_k:.3f}")
                        st.metric("Context Recall", f"{m.retrieval.context_recall:.3f}")
                        st.metric("Context Precision", f"{m.retrieval.context_precision:.3f}")

                    if m.failure_modes:
                        fm_names = [fm.value for fm in m.failure_modes]
                        st.markdown("**Failure Modes:** " + " · ".join(f"`{fm}`" for fm in fm_names))
                    if m.warnings:
                        for w in m.warnings:
                            st.warning(w)

                if record.errors:
                    for err in record.errors:
                        st.error(err)

# ── Page: Metrics Leaderboard ─────────────────────────────────────────────────

elif page == "🏆 Metrics Leaderboard":
    st.title("🏆 Metrics Leaderboard")

    if df.empty:
        st.info("No run data available.")
    else:
        comp = ExperimentComparison(df)
        leaderboard = comp.leaderboard()

        st.subheader("Experiment Rankings")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        if "experiment_id" in df.columns and len(df["experiment_id"].unique()) > 1:
            st.subheader("Score Distributions")
            metric_to_show = st.selectbox(
                "Select metric",
                ["overall_score", "groundedness", "answer_relevance",
                 "citation_coverage", "context_recall"],
            )
            if metric_to_show in df.columns:
                agg = df.groupby("experiment_id")[metric_to_show].mean().sort_values(ascending=False)
                st.bar_chart(agg)

        st.subheader("Detailed Statistics")
        stats = comp.summary_statistics()
        if not stats.empty:
            st.dataframe(stats, use_container_width=True)

# ── Page: Retrieval Analysis ──────────────────────────────────────────────────

elif page == "📡 Retrieval Analysis":
    st.title("📡 Retrieval Analysis")

    if df.empty:
        st.info("No data available.")
    else:
        ret_metrics = ["hit_at_k", "reciprocal_rank", "context_precision", "context_recall"]
        available_ret = [c for c in ret_metrics if c in df.columns]

        if available_ret:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Retrieval Metrics Summary")
                ret_summary = df[available_ret].describe().T.round(4)
                st.dataframe(ret_summary[["mean", "std", "min", "max"]], use_container_width=True)

            with col2:
                st.subheader("Hit@K by Experiment")
                if "experiment_id" in df.columns and "hit_at_k" in df.columns:
                    hit_agg = df.groupby("experiment_id")["hit_at_k"].mean().sort_values(ascending=False)
                    st.bar_chart(hit_agg)

            st.subheader("Context Recall vs Answer Quality")
            if "context_recall" in df.columns and "overall_score" in df.columns:
                scatter_data = df[["context_recall", "overall_score", "example_id"]].dropna()
                st.scatter_chart(scatter_data.set_index("example_id")[["context_recall", "overall_score"]])

# ── Page: Failure Analysis ────────────────────────────────────────────────────

elif page == "⚠️ Failure Analysis":
    st.title("⚠️ Failure Analysis")
    st.markdown(
        "Analyzes systematic failure patterns in RAG pipeline outputs. "
        "Failure modes are not mutually exclusive."
    )

    if df.empty:
        st.info("No run data available.")
    else:
        if "failure_modes" not in df.columns:
            st.info("No failure mode data found.")
        else:
            from collections import Counter

            # Overall failure distribution
            all_modes: list[str] = []
            for fm in df["failure_modes"].dropna():
                all_modes.extend([m.strip() for m in str(fm).split("|") if m.strip()])

            mode_counts = Counter(all_modes)
            total = len(df)

            st.subheader("Failure Mode Frequency")
            fm_df = pd.DataFrame(
                [{"failure_mode": k, "count": v, "rate (%)": round(v / total * 100, 1)}
                 for k, v in mode_counts.most_common()]
            )
            st.dataframe(fm_df, use_container_width=True, hide_index=True)
            st.bar_chart(fm_df.set_index("failure_mode")["count"])

            # Failure mode definitions
            with st.expander("ℹ️ Failure Mode Definitions"):
                st.markdown("""
| Failure Mode | Definition |
|---|---|
| `unsupported_answer` | Answer makes claims not grounded in retrieved context (groundedness < 0.30) |
| `weak_retrieval` | Relevant documents were not retrieved (context_recall < 0.30) |
| `missing_citation` | Answerable question answered without citing expected documents |
| `incorrect_abstention` | System abstained on a question it should have answered |
| `overconfident_answer` | System answered a question that has no relevant information in corpus |
| `incomplete_answer` | Answer is too short (< 10 words) for a non-trivial question |
| `none` | No failure modes detected |
                """)

            # Bottom runs by failure category
            if "experiment_id" in df.columns:
                st.subheader("Failure Modes by Experiment")
                comp = ExperimentComparison(df)
                fm_pivot = comp.failure_mode_summary()
                if not fm_pivot.empty:
                    st.dataframe(fm_pivot, use_container_width=True, hide_index=True)

# ── Page: Experiment Comparison ───────────────────────────────────────────────

elif page == "🧪 Experiment Comparison":
    st.title("🧪 Experiment Comparison")

    if df.empty or "experiment_id" not in df.columns:
        st.info("No experiment data available.")
    else:
        experiments = sorted(df["experiment_id"].unique().tolist())

        if len(experiments) < 2:
            st.info(f"Only one experiment found ({experiments[0]}). Run more experiments to compare.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                baseline = st.selectbox("Baseline experiment", experiments, index=0)
            with col2:
                others = [e for e in experiments if e != baseline]
                comparison = st.selectbox("Comparison experiment", others, index=0)

            comp = ExperimentComparison(df)

            st.subheader("Leaderboard")
            lb = comp.leaderboard()
            st.dataframe(lb, use_container_width=True, hide_index=True)

            st.subheader(f"Delta: {comparison} vs {baseline}")
            delta = comp.delta_table(baseline, comparison)
            if not delta.empty:
                delta["better"] = delta["delta"].apply(
                    lambda x: "✅" if x > 0 else ("🔴" if x < 0 else "—")
                )
                st.dataframe(delta, use_container_width=True, hide_index=True)

            st.subheader("Per-Example Comparison (overall_score)")
            per_ex = comp.per_example_comparison(baseline, comparison, "overall_score")
            if not per_ex.empty:
                st.dataframe(per_ex, use_container_width=True, hide_index=True)
