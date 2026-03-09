# 🔬 LLM Evals & Observability Lab

> **A production-grade evaluation and observability framework for RAG and agentic LLM systems.**  
> Runs fully without paid APIs. Designed for rigorous measurement, not toy demos.

---

## Why This Project Exists

Most LLM demos show you a chatbot. This project shows you **how to know if your LLM system is actually working**.

Production LLM applications fail in subtle, non-obvious ways:
- The retriever fetches irrelevant context → the answer looks plausible but is wrong
- The generator cites documents it never used → hallucination masquerading as grounding
- Abstention is either too eager (ignoring real answers) or absent entirely
- Latency and cost vary 10× across configurations with no visibility into why

This repository demonstrates how to **measure, trace, and systematically improve** a RAG pipeline using:
- interpretable, documented evaluation metrics
- full per-run observability traces
- reproducible experiment comparison
- failure mode taxonomy and detection

---

## What Makes This Different from a Demo

| Typical LLM Demo | This Repository |
|---|---|
| Single example, manually inspected | 20-example evaluation set with reference answers |
| No metrics beyond "looks good" | 12+ metrics across retrieval and answer quality |
| Black-box output | Full traces: query → chunks → prompt → answer → scores |
| One configuration | 5 experiment configurations compared head-to-head |
| Requires API keys | Runs fully locally with TF-IDF fallback |
| No tests | 40+ pytest tests covering all modules |

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐ │
│  │ Retriever│───►│  Prompt  │───►│    Generator      │ │
│  │ (TF-IDF  │    │  Builder │    │  (Local/OpenAI)   │ │
│  │  or ST)  │    │ Baseline │    │                   │ │
│  └──────────┘    │ Grounded │    └───────────────────┘ │
│       ▲          └──────────┘             │             │
│       │                                   │             │
│  ┌──────────┐                    ┌────────▼──────────┐ │
│  │  Chunk   │                    │    Evaluator      │ │
│  │  Index   │                    │  (12+ metrics)    │ │
│  └──────────┘                    └────────────────────┘│
└─────────────────────────────────────────────────────────┘
    │                                        │
    ▼                                        ▼
┌──────────────────┐               ┌──────────────────┐
│  Observability   │               │    Run Store     │
│  Tracer          │──────────────►│  JSON + CSV      │
│  (per-step time) │               │  results/runs/   │
└──────────────────┘               └──────────────────┘
                                            │
                                   ┌────────▼──────────┐
                                   │  Streamlit         │
                                   │  Dashboard         │
                                   └───────────────────┘
```

### Core Components

| Component | Location | Purpose |
|---|---|---|
| **Corpus** | `data/raw/corpus.json` | 10 NovaSaaS knowledge base documents |
| **Chunks** | `data/processed/chunks.json` | Sliding-window chunks with metadata |
| **Eval Set** | `data/eval/eval_set.json` | 20 graded evaluation examples |
| **Embedder** | `src/.../retrieval/embedder.py` | TF-IDF (default) or sentence-transformers |
| **Index** | `src/.../retrieval/index.py` | Dense numpy cosine similarity index |
| **Retriever** | `src/.../retrieval/retriever.py` | Top-K retrieval with score normalization |
| **Prompts** | `src/.../generation/prompts.py` | Baseline and citation-first templates |
| **Generator** | `src/.../generation/generator.py` | Local heuristic + OpenAI-compatible |
| **Evaluator** | `src/.../evaluation/evaluator.py` | 12-metric evaluation engine |
| **Tracer** | `src/.../observability/tracer.py` | Per-run trace accumulation |
| **Run Store** | `src/.../observability/run_store.py` | JSON persistence + CSV summaries |
| **Dashboard** | `app/dashboard.py` | 9-section Streamlit dashboard |

---

## Evaluation Metrics

All metrics are in **[0, 1]** unless noted. Higher is better except where marked.

### Answer Quality Metrics

| Metric | Method | Notes |
|---|---|---|
| `answer_relevance_score` | Word-overlap (question ↔ answer) | Proxy for topical alignment |
| `groundedness_score` | Sentence attribution + chunk overlap | Fraction of answer supported by context |
| `citation_coverage_score` | Set intersection (cited ↔ expected docs) | Measures source traceability |
| `exact_match_proxy` | Token-level F1 (SQuAD-style) | Against reference answer |
| `faithfulness_proxy` | Answer ↔ context Jaccard overlap | Proxy for NLI faithfulness |
| `hallucination_risk_score` | 1 - groundedness + length penalty | ⚠️ Higher = more risk |
| `abstention_quality_score` | Correctness of abstain/answer decision | 1.0 if unanswerable+abstained |
| `overall_score` | Weighted composite (configurable) | Primary ranking signal |

### Retrieval Quality Metrics

| Metric | Definition |
|---|---|
| `hit_at_k` | ≥1 expected doc in top-K retrieved chunks |
| `reciprocal_rank` | 1/rank of first relevant doc (MRR-style) |
| `context_precision` | Fraction of retrieved chunks whose doc is relevant |
| `context_recall` | Fraction of expected docs covered by retrieved set |

### Operational Metrics

| Metric | Notes |
|---|---|
| `latency_ms` | Wall-clock time for full pipeline pass |
| `estimated_cost_usd` | $0 in local mode; non-zero for API backends |

### Overall Score Weights (configurable in `configs/eval.yaml`)

```yaml
score_weights:
  answer_relevance: 0.20
  groundedness:     0.25   # highest weight — most critical for safety
  citation_coverage: 0.15
  context_precision: 0.15
  context_recall:   0.15
  faithfulness_proxy: 0.10
```

> **Transparency note:** All metrics are heuristic proxies. Groundedness and faithfulness use word-overlap and sentence attribution rather than NLI models. This is intentional — the repo runs without GPUs or API costs. For production use, consider replacing with a dedicated NLI scorer or LLM-as-a-judge. Methodology details in [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md).

---

## Failure Mode Taxonomy

The system detects and classifies the following failure patterns:

| Failure Mode | Detection | Meaning |
|---|---|---|
| `unsupported_answer` | groundedness < 0.30 | Answer not grounded in retrieved context |
| `weak_retrieval` | context_recall < 0.30 | Expected documents not retrieved |
| `missing_citation` | citation_coverage < 0.20 | Answerable but no source cited |
| `incorrect_abstention` | abstained + is_answerable | Gave up when answer existed |
| `overconfident_answer` | answered + not is_answerable | Answered when no evidence available |
| `incomplete_answer` | answer_words < 10 | Truncated or trivially short response |

---

## Quickstart

### Prerequisites

- Python 3.11+
- No API keys required for default operation

### Installation

```bash
# Clone or unzip the repository
cd llm-evals-observability-lab

# Install dependencies
pip install -r requirements.txt

# Optional: install as editable package
pip install -e .
```

### Run the Full Pipeline

```bash
# Step 1: Prepare corpus and chunks
python scripts/prepare_data.py

# Step 2: Build evaluation dataset
python scripts/build_eval_set.py

# Step 3: Run baseline evaluation
python scripts/run_baseline.py

# Step 4: Run multi-experiment comparison
python scripts/run_experiments.py

# Step 5: Aggregate results and generate figures
python scripts/analyze_results.py

# Step 6: Launch the dashboard
streamlit run app/dashboard.py
```

### Run Tests

```bash
pytest
# With coverage:
pytest --cov=src/llm_evals_lab --cov-report=term-missing
```

---

## Configuration

All configuration lives in `configs/`. No environment variables required for local operation.

### `configs/retrieval.yaml`

```yaml
retrieval:
  embedding_backend: "tfidf"   # or "sentence-transformers"
  top_k: 5
chunking:
  strategy: "sliding_window"
  chunk_size: 200
  chunk_overlap: 40
```

### `configs/eval.yaml`

```yaml
eval:
  score_weights:
    groundedness: 0.25
    answer_relevance: 0.20
    ...
  thresholds:
    groundedness_pass: 0.50
    hallucination_risk_warn: 0.60
```

### Optional API integration

Copy `.env.example` to `.env` and set:

```env
OPENAI_API_KEY=sk-...
GENERATION_BACKEND=openai
OPENAI_MODEL=gpt-4o-mini
```

The pipeline automatically uses the API when configured and falls back to local mode on failure.

---

## Experiment Configurations

`run_experiments.py` compares these configurations:

| Experiment ID | Description |
|---|---|
| `baseline_k5` | TF-IDF retriever, top-5, concise QA prompt |
| `baseline_k3` | TF-IDF retriever, top-3, concise QA prompt |
| `grounded_k5` | TF-IDF retriever, top-5, citation-first grounded prompt |
| `grounded_k3` | TF-IDF retriever, top-3, citation-first grounded prompt |
| `high_abstain_k5` | Higher abstention threshold (conservative answering) |

---

## Repository Structure

```
llm-evals-observability-lab/
├── configs/                   # YAML configuration files
├── data/
│   ├── raw/corpus.json        # Source documents (NovaSaaS KB)
│   ├── processed/chunks.json  # Processed chunks + index
│   └── eval/eval_set.json     # 20-example evaluation set
├── docs/                      # Architecture, methodology, roadmap
├── notebooks/                 # 4 analysis notebooks
├── app/dashboard.py           # Streamlit dashboard (9 sections)
├── scripts/                   # 5 runnable pipeline scripts
├── src/llm_evals_lab/         # Core Python package
│   ├── config.py              # Config loading and resolution
│   ├── schemas.py             # Pydantic data models
│   ├── utils.py               # Shared utilities
│   ├── data/                  # Corpus loader, chunking, eval set
│   ├── retrieval/             # Embedder, index, retriever
│   ├── generation/            # Prompts, generator, RAG pipeline
│   ├── evaluation/            # All metric implementations
│   ├── observability/         # Tracer, logger, run store
│   ├── experiments/           # Comparison and leaderboard
│   └── visualization/         # Plotly and Matplotlib charts
├── tests/                     # 40+ pytest tests
└── results/
    ├── runs/                  # Per-run JSON traces
    ├── tables/                # CSV summaries and leaderboards
    └── figures/               # Generated charts
```

---

## Dashboard Sections

Launch with `streamlit run app/dashboard.py`:

1. **Overview** — KPIs, recent runs, score by experiment
2. **Corpus Explorer** — Browse source documents and chunks
3. **Eval Set Explorer** — Browse eval examples with filtering
4. **Run Explorer** — Filter and sort run history
5. **Trace Viewer** — Full trace inspection for any run
6. **Metrics Leaderboard** — Ranked experiment comparison
7. **Retrieval Analysis** — Hit@K, MRR, precision, recall
8. **Failure Analysis** — Mode distribution with definitions
9. **Experiment Comparison** — Delta tables and per-example comparison

---

## Sample Artifacts

After running the full pipeline:

```
results/
├── runs/
│   └── <uuid>.json             # Full trace per run (100 files for 5 experiments × 20 examples)
├── tables/
│   ├── run_summary.csv         # Flat table of all run metrics
│   ├── baseline_results.csv    # Baseline experiment results
│   ├── leaderboard.csv         # Experiment leaderboard
│   ├── failure_mode_analysis.csv
│   └── hardest_examples.csv
└── figures/
    ├── overall_score_bar.png
    └── metric_heatmap.png
```

---

## Limitations

This is an evaluation framework, not a production LLM system. Known limitations:

1. **Local generator is heuristic.** The `LocalGenerator` uses word-overlap and sentence extraction rather than a language model. It produces evaluation-worthy outputs but not human-quality prose. Replace with `generation_backend=openai` for realistic answers.

2. **Metrics are heuristic proxies.** Groundedness and faithfulness use word-overlap, not neural entailment. They are interpretable and cost-free, but less accurate than NLI-based scorers or LLM-as-a-judge.

3. **Corpus is synthetic.** The NovaSaaS knowledge base is fictional. Retrieval quality will differ on real-world corpora with more ambiguity, jargon, and length variation.

4. **No vector database.** The index is in-memory numpy. For production scale (>100k chunks), integrate FAISS, Chroma, or Qdrant.

5. **No re-ranking.** The retriever returns raw cosine similarity results without a cross-encoder re-ranking step.

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full roadmap.

Near-term priorities:
- [ ] Cross-encoder re-ranking support
- [ ] LLM-as-a-judge evaluation mode (optional, needs API)
- [ ] FAISS integration for larger corpora
- [ ] Pytest-based regression gate (alert on score drops)
- [ ] Multi-document synthesis eval examples
- [ ] Structured output evaluation (JSON, tables)

---

## Why This is Interesting for Applied AI / Senior ML Roles

This project demonstrates:

- **Evaluation rigor** — Not "the model said something reasonable" but a 12-metric measurement framework with documented methodology and explicit limitations
- **Observability mindset** — Every run produces a full trace: every retrieved chunk, every prompt, every metric, every warning
- **Experiment discipline** — Configurations compared head-to-head with delta tables and statistical summaries
- **Production thinking** — Graceful fallbacks, configurable thresholds, idempotent storage, cost tracking
- **Systems architecture** — Clean separation between retrieval, generation, evaluation, and observability layers
- **Failure analysis** — Systematic taxonomy of what can go wrong and how to detect it

These are the skills that matter when moving from demo to production: knowing when your system is broken, why it's broken, and how to measure improvement.

---

## License

MIT License — see `LICENSE` for details.

---

## Contributing

Issues and PRs welcome. For major changes, please open a discussion first.

```bash
# Development setup
pip install -e ".[dev]"
pytest
```
