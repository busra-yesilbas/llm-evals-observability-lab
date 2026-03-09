# Roadmap

This document outlines the planned evolution of the LLM Evals & Observability Lab.

---

## Current State (v0.1.0)

- ✅ 10-document synthetic NovaSaaS knowledge base
- ✅ Sliding-window chunking with configurable overlap
- ✅ TF-IDF retriever (default) with sentence-transformers optional backend
- ✅ Local heuristic answer generator (no API required)
- ✅ OpenAI-compatible API generator with fallback
- ✅ 12 evaluation metrics (retrieval + answer quality + failure modes)
- ✅ Full trace persistence per run (JSON + CSV)
- ✅ 5-configuration experiment comparison suite
- ✅ 9-section Streamlit dashboard
- ✅ 4 analysis notebooks
- ✅ 40+ pytest tests

---

## Near-Term (v0.2.0)

### Retrieval Improvements
- [ ] **Cross-encoder re-ranking** — Add a second-stage re-ranker using `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers). Improves precision without requiring embedding re-training.
- [ ] **FAISS integration** — Replace the numpy flat index with FAISS IVF for efficient retrieval on large corpora (>100k chunks).
- [ ] **Hybrid retrieval** — BM25 + dense embeddings with reciprocal rank fusion (RRF). Captures both lexical and semantic similarity.
- [ ] **Retrieval diversity filtering** — MMR (Maximal Marginal Relevance) to reduce redundant chunks in context.

### Evaluation Improvements
- [ ] **LLM-as-a-judge mode** — Optional evaluation pass using an LLM to score faithfulness and relevance. Gated behind `GENERATION_BACKEND=openai` config.
- [ ] **NLI-based groundedness** — Replace word-overlap groundedness with a lightweight cross-encoder NLI classifier.
- [ ] **Expanded eval set** — 50 examples covering more ambiguous and multi-hop cases.
- [ ] **Regression alerting** — Pytest plugin that fails if any experiment's `overall_score` drops >5% from the stored baseline.

### Observability Improvements
- [ ] **OpenTelemetry export** — Export spans from the tracer to OTLP endpoints (Jaeger, Tempo, Honeycomb).
- [ ] **Prompt diff tracking** — Store diffs between prompt versions to correlate prompt changes with metric changes.
- [ ] **Cost tracking by token** — Accurate token counting using `tiktoken` for OpenAI models.

---

## Medium-Term (v0.3.0)

### Agentic Evaluation
- [ ] **Tool-use pipeline** — Extend to evaluate agents with tool calls (search, calculator, code interpreter). Trace tool invocations alongside retrieval.
- [ ] **Multi-turn evaluation** — Support multi-turn RAG conversations and measure cross-turn consistency.
- [ ] **Agent failure taxonomy** — Extended failure modes for agents: tool hallucination, infinite loops, goal drift.

### Scale and Performance
- [ ] **Async pipeline** — Async retrieval and generation for batch evaluation throughput.
- [ ] **Distributed experiment runner** — Run experiments in parallel with multiprocessing or Ray.
- [ ] **Corpus expansion** — Support 1000+ document corpora with incremental index updates.

### Corpus and Eval Set
- [ ] **Real-world corpus adapters** — Loaders for Common Crawl subsets, Wikipedia, and arXiv abstracts.
- [ ] **Eval set generator** — Automated question generation from corpus using an LLM with human review workflow.
- [ ] **Adversarial examples** — Prompt injection, context confusion, and adversarial retrieval tests.

---

## Long-Term (v1.0.0)

### Production Integration
- [ ] **REST API** — Expose pipeline and eval endpoints as a FastAPI service for integration testing.
- [ ] **CI/CD integration** — GitHub Actions workflow that runs the eval suite on every PR and blocks merge on regression.
- [ ] **Grafana/Prometheus dashboards** — Production monitoring metrics exported from the run store.

### Research Features
- [ ] **Calibration analysis** — Do the proxy metrics correlate with human judgments? Add a human eval comparison mode.
- [ ] **Counterfactual analysis** — "What if we had retrieved doc X instead?" — counterfactual context substitution.
- [ ] **Ensemble eval** — Average multiple evaluation methods to reduce single-metric variance.

---

## Non-Goals

The following are explicitly out of scope for this project:

- Building a production-ready LLM application or chatbot
- Fine-tuning or training embedding models
- Implementing a full agentic framework (use LangGraph or CrewAI for that)
- Replacing purpose-built eval frameworks like RAGAS for production use

This project's value is demonstrating evaluation and observability methodology, not building a complete LLM platform.

---

## Contributing

If you'd like to contribute to any of these items, open an issue to discuss the approach before submitting a PR. New metrics should include:
1. Clear mathematical definition
2. Documented assumptions and limitations
3. Unit tests
4. Update to `docs/evaluation_methodology.md`
