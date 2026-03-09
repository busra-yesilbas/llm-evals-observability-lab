"""
Prompt templates for the LLM Evals Lab.

Two strategies are implemented:
  BASELINE  — concise QA prompt, retrieved context appended as-is.
  GROUNDED  — citation-first format that explicitly asks for source references
               and instructs the model to abstain when evidence is insufficient.

Both strategies are available for the local answer generator (no API required).
"""

from __future__ import annotations

from string import Template
from typing import Any

from llm_evals_lab.schemas import PromptStrategy, RetrievedChunk


# ── Prompt templates ──────────────────────────────────────────────────────────

BASELINE_TEMPLATE = Template(
    """\
You are a helpful support assistant for NovaSaaS, a B2B SaaS platform.
Answer the user's question using ONLY the information provided in the context below.
Be concise and factual.

CONTEXT:
$context

QUESTION: $question

ANSWER:"""
)

GROUNDED_TEMPLATE = Template(
    """\
You are a precise support assistant for NovaSaaS. Your answers must be \
fully supported by the provided context passages.

Instructions:
- Answer using ONLY information found in the context.
- After your answer, list the source chunk IDs you used as [Source: chunk_id].
- If the context does not contain enough information to answer, respond with:
  "I cannot answer this question based on the available documentation."
- Do NOT speculate or add information beyond the context.

CONTEXT PASSAGES:
$context

QUESTION: $question

ANSWER (with citations):"""
)


# ── Prompt builder ────────────────────────────────────────────────────────────


def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    strategy: PromptStrategy = PromptStrategy.BASELINE,
    max_context_words: int = 800,
) -> str:
    """
    Assemble a prompt string from a question and retrieved chunks.

    Parameters
    ----------
    question : str
        The user question.
    retrieved_chunks : list[RetrievedChunk]
        Chunks retrieved from the index.
    strategy : PromptStrategy
        Which prompt template to use.
    max_context_words : int
        Truncate context to this many words to avoid overly long prompts.

    Returns
    -------
    str
        The fully assembled prompt string.
    """
    context_parts: list[str] = []
    total_words = 0

    for chunk in retrieved_chunks:
        chunk_words = chunk.chunk_text.split()
        if total_words + len(chunk_words) > max_context_words:
            remaining = max_context_words - total_words
            if remaining > 20:
                excerpt = " ".join(chunk_words[:remaining])
                if strategy == PromptStrategy.GROUNDED:
                    context_parts.append(f"[{chunk.chunk_id}] {excerpt}...")
                else:
                    context_parts.append(excerpt + "...")
            break

        if strategy == PromptStrategy.GROUNDED:
            context_parts.append(f"[{chunk.chunk_id}] {chunk.chunk_text}")
        else:
            context_parts.append(chunk.chunk_text)

        total_words += len(chunk_words)

    context_str = "\n\n".join(context_parts) if context_parts else "(No relevant context found.)"

    if strategy == PromptStrategy.BASELINE:
        return BASELINE_TEMPLATE.substitute(context=context_str, question=question)
    elif strategy == PromptStrategy.GROUNDED:
        return GROUNDED_TEMPLATE.substitute(context=context_str, question=question)
    else:
        raise ValueError(f"Unknown prompt strategy: {strategy}")


def format_context_for_display(chunks: list[RetrievedChunk]) -> str:
    """Human-readable context display (for notebooks/dashboard)."""
    lines: list[str] = []
    for c in chunks:
        lines.append(f"--- [{c.chunk_id}] (score={c.score:.3f}) ---")
        lines.append(c.chunk_text)
        lines.append("")
    return "\n".join(lines)
