"""
Answer generator for the LLM Evals Lab.

Provides a local fallback generator (no API required) and an optional
OpenAI-compatible API generator. The local generator synthesizes answers
from retrieved context using heuristic extraction and templating —
sufficient for evaluation and observability demonstrations.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

from llm_evals_lab.schemas import GeneratedAnswer, PromptStrategy, RetrievedChunk
from llm_evals_lab.utils import normalize_text, word_overlap_score

logger = logging.getLogger(__name__)

# Phrases that indicate an abstention response
_ABSTAIN_PHRASES = [
    "cannot answer",
    "not enough information",
    "not mentioned",
    "not available",
    "not in the documentation",
    "no information",
    "insufficient context",
    "i don't know",
    "unclear from",
]

# Threshold below which the local generator abstains
_ABSTAIN_OVERLAP_THRESHOLD = 0.03


class BaseGenerator(ABC):
    """Abstract base for all answer generators."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        retrieved_chunks: list[RetrievedChunk],
        question: str,
        strategy: PromptStrategy = PromptStrategy.BASELINE,
    ) -> GeneratedAnswer:
        """Generate an answer given a prompt and retrieved context."""


class LocalGenerator(BaseGenerator):
    """
    Heuristic answer generator that does NOT call any external API.

    Strategy
    --------
    1. Identify the most relevant sentence(s) from retrieved chunks using
       word-overlap with the question.
    2. Assemble a compact answer from the top sentences.
    3. If the best overlap score is below threshold, abstain gracefully.
    4. For the GROUNDED strategy, append synthetic citations.

    This produces evaluation-ready answers that exercise the full metrics
    pipeline without requiring paid API access.
    """

    def __init__(
        self,
        abstain_threshold: float = _ABSTAIN_OVERLAP_THRESHOLD,
        max_answer_sentences: int = 4,
    ) -> None:
        self.abstain_threshold = abstain_threshold
        self.max_answer_sentences = max_answer_sentences

    def generate(
        self,
        prompt: str,
        retrieved_chunks: list[RetrievedChunk],
        question: str,
        strategy: PromptStrategy = PromptStrategy.BASELINE,
    ) -> GeneratedAnswer:
        if not retrieved_chunks:
            return self._abstain(strategy)

        # Score each chunk by relevance to the question
        scored_chunks = [
            (chunk, word_overlap_score(question, chunk.chunk_text))
            for chunk in retrieved_chunks
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        best_score = scored_chunks[0][1]
        if best_score < self.abstain_threshold:
            logger.debug(
                "Best overlap score %.3f < threshold %.3f — abstaining",
                best_score,
                self.abstain_threshold,
            )
            return self._abstain(strategy)

        # Extract top sentences from the most relevant chunks
        answer_sentences: list[str] = []
        cited_chunk_ids: list[str] = []
        cited_doc_ids: list[str] = []

        for chunk, score in scored_chunks:
            if len(answer_sentences) >= self.max_answer_sentences:
                break
            sentences = self._split_sentences(chunk.chunk_text)
            for sent in sentences:
                sent_score = word_overlap_score(question, sent)
                if sent_score > 0.0 and len(answer_sentences) < self.max_answer_sentences:
                    answer_sentences.append(sent.strip())
                    if chunk.chunk_id not in cited_chunk_ids:
                        cited_chunk_ids.append(chunk.chunk_id)
                    if chunk.doc_id not in cited_doc_ids:
                        cited_doc_ids.append(chunk.doc_id)

        if not answer_sentences:
            # Fall back to first sentence of best chunk
            sentences = self._split_sentences(scored_chunks[0][0].chunk_text)
            answer_sentences = sentences[:2]
            cited_chunk_ids = [scored_chunks[0][0].chunk_id]
            cited_doc_ids = [scored_chunks[0][0].doc_id]

        answer_text = " ".join(answer_sentences)

        # For grounded strategy, append citation markers
        if strategy == PromptStrategy.GROUNDED and cited_chunk_ids:
            citation_str = ", ".join(f"[Source: {cid}]" for cid in cited_chunk_ids)
            answer_text = f"{answer_text} {citation_str}"

        return GeneratedAnswer(
            answer_text=answer_text,
            cited_chunk_ids=cited_chunk_ids,
            cited_doc_ids=cited_doc_ids,
            abstained=False,
            prompt_strategy=strategy,
        )

    def _abstain(self, strategy: PromptStrategy) -> GeneratedAnswer:
        text = "I cannot answer this question based on the available documentation."
        return GeneratedAnswer(
            answer_text=text,
            cited_chunk_ids=[],
            cited_doc_ids=[],
            abstained=True,
            prompt_strategy=strategy,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Lightweight sentence splitter."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if len(s.split()) >= 5]


class OpenAIGenerator(BaseGenerator):
    """
    OpenAI-compatible API generator.

    Requires OPENAI_API_KEY in environment. Falls back to LocalGenerator
    if the API call fails.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        fallback: Optional[BaseGenerator] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback = fallback or LocalGenerator()

    def generate(
        self,
        prompt: str,
        retrieved_chunks: list[RetrievedChunk],
        question: str,
        strategy: PromptStrategy = PromptStrategy.BASELINE,
    ) -> GeneratedAnswer:
        try:
            return self._call_api(prompt, strategy)
        except Exception as exc:
            logger.warning("OpenAI API call failed (%s) — using local fallback.", exc)
            return self.fallback.generate(prompt, retrieved_chunks, question, strategy)

    def _call_api(self, prompt: str, strategy: PromptStrategy) -> GeneratedAnswer:
        try:
            import openai  # type: ignore
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        raw_text: str = response.choices[0].message.content or ""

        # Parse citation markers from grounded strategy output
        cited_chunk_ids = re.findall(r"\[Source:\s*([^\]]+)\]", raw_text)
        abstained = any(phrase in raw_text.lower() for phrase in _ABSTAIN_PHRASES)

        return GeneratedAnswer(
            answer_text=raw_text.strip(),
            cited_chunk_ids=cited_chunk_ids,
            cited_doc_ids=[],
            abstained=abstained,
            prompt_strategy=strategy,
        )


def get_generator(
    backend: str = "local",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    abstain_threshold: float = _ABSTAIN_OVERLAP_THRESHOLD,
) -> BaseGenerator:
    """
    Factory: return the appropriate generator backend.

    Parameters
    ----------
    backend : str
        "local" (default, no API) or "openai".
    model : str, optional
        Model name for API backends.
    api_key : str, optional
        API key for external backends.
    base_url : str, optional
        Custom base URL for OpenAI-compatible APIs.
    abstain_threshold : float
        Overlap threshold below which the local generator abstains.
    """
    if backend == "openai":
        return OpenAIGenerator(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            base_url=base_url,
            fallback=LocalGenerator(abstain_threshold=abstain_threshold),
        )
    return LocalGenerator(abstain_threshold=abstain_threshold)
