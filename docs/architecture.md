# Architecture

## System Overview

The LLM Evals & Observability Lab is organized as a layered system where each layer has a single, well-defined responsibility. The layers communicate only through Pydantic schema objects, making it easy to swap implementations.

```
┌──────────────────────────────────────────────────────────┐
│                    Interface Layer                        │
│         Streamlit Dashboard  ·  CLI Scripts              │
└──────────────────────┬───────────────────────────────────┘
                       │ RunRecord, DataFrames
┌──────────────────────▼───────────────────────────────────┐
│               Orchestration Layer                         │
│            RAGPipeline  ·  ExperimentRunner              │
└──────┬───────────────┬───────────────────┬───────────────┘
       │               │                   │
┌──────▼──────┐  ┌─────▼──────┐  ┌────────▼──────────────┐
│  Retrieval  │  │ Generation │  │     Evaluation         │
│  Embedder   │  │  Prompts   │  │  Answer Quality        │
│  Index      │  │  Generator │  │  Retrieval Metrics     │
│  Retriever  │  │            │  │  Groundedness          │
└─────────────┘  └────────────┘  └───────────────────────┘
       │               │                   │
┌──────▼───────────────▼───────────────────▼──────────────┐
│                Observability Layer                        │
│         Tracer  ·  RunStore  ·  StructuredLogger         │
└─────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                   Storage Layer                           │
│      results/runs/*.json  ·  results/tables/*.csv        │
└──────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `llm_evals_lab.config`
Loads and merges three YAML config files (`app.yaml`, `eval.yaml`, `retrieval.yaml`) and environment variables into a single `LabConfig` object. All path resolution is rooted at the project root, making scripts portable.

### `llm_evals_lab.schemas`
Pydantic v2 models for all data contracts. Key types:
- `SourceDocument` / `DocumentChunk` — corpus data
- `EvalExample` — evaluation ground truth
- `RetrievedChunk` / `GeneratedAnswer` — pipeline outputs
- `RetrievalMetrics` / `AnswerMetrics` / `RunMetrics` — evaluation results
- `RunRecord` — complete trace (inputs + outputs + metrics)

### `llm_evals_lab.data`
- `CorpusLoader`: JSON-based read/write for documents and chunks
- `Chunker`: Sliding-window, sentence-boundary, and fixed-size chunking strategies
- `EvalSetLoader`: Read/write for evaluation examples; `build_synthetic_eval_set()` factory

### `llm_evals_lab.retrieval`
- `BaseEmbedder` ABC with `TFIDFEmbedder` (sklearn) and `SentenceTransformerEmbedder` (optional)
- `get_embedder()` factory: returns best available backend, falling back gracefully to TF-IDF
- `ChunkIndex`: Dense numpy matrix for cosine similarity search (dot product of L2-normalised vectors)
- `Retriever`: High-level retrieval interface; handles index persistence

### `llm_evals_lab.generation`
- `build_prompt()`: Assembles prompt from question + retrieved chunks, supporting BASELINE and GROUNDED templates
- `LocalGenerator`: Heuristic answer generator using sentence extraction (no API)
- `OpenAIGenerator`: OpenAI-compatible API generator with fallback to local
- `get_generator()` factory
- `RAGPipeline`: Orchestrates the full retrieval → prompt → generation → evaluation → tracing cycle

### `llm_evals_lab.evaluation`
- `retrieval_metrics.py`: Hit@K, reciprocal rank, context precision, context recall
- `groundedness.py`: Sentence attribution score + chunk overlap score
- `answer_quality.py`: Answer relevance, citation coverage, F1 exact match proxy, faithfulness proxy, abstention quality
- `metrics.py`: Failure mode detection, cost estimation, weighted composite score
- `Evaluator`: Orchestrates all metrics for a RunRecord+EvalExample pair

### `llm_evals_lab.observability`
- `Tracer`: Mutable accumulator built progressively during a pipeline run; produces `RunRecord`
- `StructuredLogger`: JSON-formatted structured log events
- `RunStore`: Persists `RunRecord` as JSON (full trace) + maintains a rolling CSV summary

### `llm_evals_lab.experiments`
- `ExperimentComparison`: Leaderboard, delta tables, per-example comparison, failure mode pivot tables
- `compare_experiments()`: Convenience wrapper for full comparison suite

### `llm_evals_lab.visualization`
- Plotly (interactive): Radar chart, bar chart, box plot, heatmap, scatter
- Matplotlib (static): Bar chart, heatmap — saved to `results/figures/`

## Data Flow

```
prepare_data.py
  └─► SourceDocument[] → Chunker → DocumentChunk[] → saved to disk

build_eval_set.py
  └─► EvalExample[] → saved to disk

run_baseline.py / run_experiments.py
  └─► for each EvalExample:
       1. Retriever.retrieve(query) → RetrievedChunk[]
       2. build_prompt(query, chunks) → str
       3. Generator.generate(prompt) → GeneratedAnswer
       4. Evaluator.evaluate(record, example) → RunMetrics
       5. RunStore.save(RunRecord) → {run_id}.json + run_summary.csv

analyze_results.py
  └─► RunStore.load_all() → DataFrame → leaderboard, failure analysis, figures
```

## Key Design Decisions

### 1. Fallback-first embedding
The repo must work without `sentence-transformers`. `get_embedder()` catches `ImportError` and returns `TFIDFEmbedder`. This ensures the full pipeline runs in any environment.

### 2. Schema-first design
All module boundaries are defined by Pydantic schemas. This provides runtime validation, auto-serialization, and makes it easy to inspect intermediate outputs in notebooks.

### 3. Idempotent run storage
`RunStore.save()` checks for duplicate `run_id` before writing to the CSV summary. This makes scripts safely re-runnable.

### 4. Config-driven thresholds
All failure mode thresholds, score weights, and cost parameters are in `configs/eval.yaml`, not hardcoded. Researchers can iterate on thresholds without modifying source code.

### 5. Trace before score
The `Tracer` accumulates state throughout the pipeline. Metrics are computed as a post-step after the trace is built, not inline. This means the trace is always valid even if metric computation fails.

## Extension Points

| To add... | Modify or extend... |
|---|---|
| New embedding backend | `BaseEmbedder` subclass in `embedder.py` |
| New generation backend | `BaseGenerator` subclass in `generator.py` |
| New evaluation metric | Add to `answer_quality.py` or `retrieval_metrics.py`, update `Evaluator` |
| New prompt strategy | Add to `PromptStrategy` enum and `prompts.py` |
| New failure mode | Add to `FailureMode` enum and `detect_failure_modes()` in `metrics.py` |
| Persistent vector DB | Replace `ChunkIndex` internals; keep the `search()` interface |
