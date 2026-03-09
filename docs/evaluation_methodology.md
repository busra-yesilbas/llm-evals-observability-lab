# Evaluation Methodology

## Overview

This document describes the evaluation framework used in the LLM Evals Lab: what each metric measures, how it is computed, and what its limitations are. We make no claim that these metrics are production-perfect — they are interpretable, cost-free proxies that provide signal without requiring GPUs or paid API access.

All metric implementations are in `src/llm_evals_lab/evaluation/`.

---

## Metric Taxonomy

We group metrics into three categories:

1. **Retrieval metrics** — Did the retriever find the right evidence?
2. **Answer quality metrics** — Is the answer correct, grounded, and complete?
3. **Operational metrics** — Latency, cost, and system-level health

---

## Retrieval Metrics

### Hit@K (`hit_at_k`)

**Definition:** Binary. 1 if any expected document appears in the top-K retrieved chunks; 0 otherwise.

**Computation:**
```python
hit = float(any(did in expected_set for did in retrieved_doc_ids))
```

**Interpretation:** Measures whether the retriever can find at least one relevant document. Low Hit@K means the retrieval step is fundamentally broken — no amount of generation quality improvement will help.

**Limitations:** Does not account for how high in the ranking the relevant document appears.

---

### Reciprocal Rank (`reciprocal_rank`)

**Definition:** 1/rank of the first relevant document in the retrieved list. 0 if none found.

**Computation:**
```python
rr = 1.0 / rank  # where rank is the 1-indexed position of first relevant doc
```

**Interpretation:** Rewards retrievers that place relevant documents near the top. A score of 0.5 means the first relevant document was at rank 2.

**Relationship to MRR:** When averaged across queries, this is Mean Reciprocal Rank (MRR). We report per-query reciprocal rank; compute the mean externally for MRR.

---

### Context Precision (`context_precision`)

**Definition:** Fraction of retrieved chunks whose parent document is in the expected set.

**Computation:**
```python
precision = relevant_retrieved_chunks / total_retrieved_chunks
```

**Interpretation:** Measures retrieval noise. Low precision means the retriever is flooding the context with irrelevant material, which increases hallucination risk and prompt length.

---

### Context Recall (`context_recall`)

**Definition:** Fraction of expected documents that appear at least once in the retrieved set.

**Computation:**
```python
recall = |expected_docs ∩ retrieved_docs| / |expected_docs|
```

**Interpretation:** Measures retrieval coverage. Low recall means evidence was missing. In multi-hop questions, recall < 1.0 means the generator lacked necessary context.

---

## Answer Quality Metrics

### Answer Relevance (`answer_relevance_score`)

**Definition:** Estimated topical alignment between the question and the generated answer.

**Computation:** Jaccard word overlap between question and answer, scaled by 4×:
```python
overlap = |words(question) ∩ words(answer)| / |words(question) ∪ words(answer)|
relevance = min(1.0, overlap * 4.0)
```

**Rationale:** Raw Jaccard is typically 0.05–0.25 for topically aligned text. The ×4 scale maps this to a more intuitive [0, 1] range. Text normalization (lowercase, remove punctuation) is applied before comparison.

**Limitations:** Does not understand semantics. A factually wrong but topically aligned answer scores the same as a correct one. For production, consider embedding-based cosine similarity or LLM-as-a-judge.

---

### Groundedness (`groundedness_score`)

**Definition:** Degree to which the generated answer is supported by the retrieved context.

**Computation (two signals combined):**

*Signal 1 — Chunk overlap:*
```python
chunk_overlap = 0.4 * mean_overlap + 0.6 * max_overlap
```
where `overlap = Jaccard(answer, chunk_text)` for each chunk.

*Signal 2 — Sentence attribution:*
For each sentence in the answer, check if any chunk has overlap ≥ 0.08:
```python
attribution = attributed_sentences / total_answer_sentences
```

*Combined:*
```python
groundedness = 0.35 * chunk_overlap + 0.65 * attribution
```

**Rationale:** Attribution (sentence-level) is a stronger signal than average overlap, so it receives higher weight. A sentence that does not overlap with any chunk at ≥8% is considered unattributed.

**Special case:** Abstained answers receive groundedness = 1.0 (they make no unsupported claims).

**Limitations:** Word overlap ≠ entailment. A sentence can share words with a chunk without being logically supported by it. True groundedness requires natural language inference (NLI). The RAGAS framework uses an LLM-as-a-judge; we use a proxy that runs without APIs.

---

### Citation Coverage (`citation_coverage_score`)

**Definition:** Fraction of expected documents that were cited in the answer.

**Computation:**
```python
coverage = |cited_doc_ids ∩ expected_doc_ids| / |expected_doc_ids|
```

**Interpretation:** Measures source traceability. Low citation coverage means the answer is not connected to evidence, even if the answer text happens to be correct.

**Note:** In local mode, citation IDs are extracted from `[Source: chunk_id]` markers or from the chunks used during answer construction. In API mode, they are parsed from the model output.

---

### Exact Match Proxy (`exact_match_proxy`)

**Definition:** Token-level F1 between the generated answer and the reference answer, following the SQuAD evaluation metric.

**Computation:**
```python
common = sum(min(pred_count[t], ref_count[t]) for t in ref_tokens)
precision = common / len(pred_tokens)
recall = common / len(ref_tokens)
f1 = 2 * precision * recall / (precision + recall)
```

Text is normalized (lowercase, remove punctuation) before tokenization.

**Interpretation:** Standard reading-comprehension metric. 1.0 = token-perfect match; 0.0 = no shared tokens. Values of 0.3–0.6 typically indicate a partially correct or differently-worded correct answer.

---

### Faithfulness Proxy (`faithfulness_proxy`)

**Definition:** Proxy for whether the answer stays within the bounds of the retrieved context.

**Computation:**
```python
combined_context = " ".join(chunk.chunk_text for chunk in retrieved_chunks)
faithfulness = min(1.0, Jaccard(answer, combined_context) * 3.5)
```

**Relationship to groundedness:** Faithfulness differs from groundedness in direction:
- **Groundedness** asks: "Is each sentence in the answer supported by some chunk?"
- **Faithfulness** asks: "Does the answer as a whole draw from the context, or does it introduce novel claims?"

**Limitations:** Same as answer relevance — word-level proxy for a semantic concept. An NLI-based approach would be more accurate.

---

### Hallucination Risk (`hallucination_risk_score`)

**Definition:** Estimated probability that the answer contains unsupported claims.

**Computation:**
```python
base_risk = 1.0 - groundedness_score

# Penalty for over-generation (answer much longer than context)
if answer_words > context_words * 1.5:
    length_penalty = min(0.2, (answer_words / context_words - 1.5) * 0.1)

risk = min(1.0, base_risk + length_penalty)
```

**Interpretation:** Higher risk → more likely the answer contains hallucinated content. The length penalty discourages answers that significantly exceed the available evidence.

**Note:** Abstained answers receive risk = 0.0.

---

### Abstention Quality (`abstention_quality_score`)

**Definition:** Measures the correctness of the abstain/answer decision.

| Situation | Score |
|---|---|
| Unanswerable + Abstained | 1.0 (correct) |
| Unanswerable + Answered | 0.0 (wrong) |
| Answerable + Abstained | 0.0 (wrong) |
| Answerable + Answered | `answer_relevance_score` |

**Rationale:** For unanswerable questions, the system should abstain rather than hallucinate. For answerable questions, abstaining is also a failure — just a different one.

---

### Overall Score (`overall_score`)

**Definition:** Weighted composite of answer and retrieval metrics.

**Default weights** (configurable in `configs/eval.yaml`):
```
overall_score = 0.20 * answer_relevance
              + 0.25 * groundedness       ← highest weight
              + 0.15 * citation_coverage
              + 0.15 * context_precision
              + 0.15 * context_recall
              + 0.10 * faithfulness_proxy
```

Groundedness receives the highest weight because unsupported claims are the primary safety concern in RAG systems.

---

## Failure Mode Detection

Failure modes are detected post-evaluation using the computed metrics:

```python
# Unsupported answer: grounded score too low
if groundedness < thresholds["unsupported_answer_threshold"]:  # 0.30
    failures.append(FailureMode.UNSUPPORTED_ANSWER)

# Weak retrieval: recall too low
if context_recall < thresholds["weak_retrieval_threshold"]:  # 0.30
    failures.append(FailureMode.WEAK_RETRIEVAL)
```

Thresholds are configurable in `configs/eval.yaml`. Multiple failure modes can apply to a single run.

---

## Evaluation Set Design

The 20-example evaluation set covers:

| Category | Examples | Key challenge |
|---|---|---|
| Factual lookup | 6 | Clear retrieval target, measurable recall |
| Policy interpretation | 3 | Multi-sentence policy analysis |
| Multi-hop synthesis | 3 | Requires evidence from 2+ documents |
| Comparison | 2 | Side-by-side plan comparison |
| Ambiguous | 2 | Vague questions testing disambiguation |
| Unanswerable | 3 | Tests abstention quality |

Unanswerable examples are critical: they catch overconfident systems that always attempt an answer regardless of evidence quality.

---

## What These Metrics Do Not Cover

- **Toxicity / safety** — Not applicable for a support KB use case, but relevant for open-domain systems
- **Factual accuracy** — We measure agreement with reference answers, not independent fact-checking
- **Coherence / fluency** — Local generator output is extractive, not fluent prose; irrelevant for metric testing
- **Multi-turn consistency** — This is single-turn RAG; dialogue coherence is out of scope

---

## Comparison with Related Frameworks

| Framework | Approach | Dependency |
|---|---|---|
| **This lab** | Heuristic word-overlap proxies | None (sklearn only) |
| [RAGAS](https://ragas.io) | LLM-as-a-judge for faithfulness/relevance | OpenAI API required |
| [TruLens](https://www.trulens.org) | NLP + LLM-based scoring | Heavyweight deps |
| [DeepEval](https://deepeval.com) | NLI + LLM-based | API or local model |
| [Evals (OpenAI)](https://github.com/openai/evals) | Flexible, template-based | Python only |

The tradeoff is explicit: this lab sacrifices metric precision for zero-dependency, offline execution. It is appropriate for rapid iteration, CI regression testing, and portfolio demonstrations.
