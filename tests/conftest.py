"""
Shared test fixtures for the LLM Evals Lab test suite.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure src/ is on the path when running pytest from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from llm_evals_lab.schemas import (
    Difficulty,
    DocumentChunk,
    EvalExample,
    GeneratedAnswer,
    PromptStrategy,
    QuestionCategory,
    RetrievedChunk,
    SourceDocument,
)


@pytest.fixture
def sample_documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            doc_id="doc_test_001",
            title="Refund Policy",
            source_type="policy",
            category="billing",
            full_text=(
                "NovaSaaS offers a 30-day money-back guarantee for all subscription plans. "
                "Refund requests must be submitted through the billing portal within 30 days. "
                "After 30 days, refunds are not available. Annual subscriptions receive prorated credit."
            ),
            metadata={"version": "1.0"},
        ),
        SourceDocument(
            doc_id="doc_test_002",
            title="Pricing Plans",
            source_type="guide",
            category="pricing",
            full_text=(
                "The Starter plan supports up to 5 user seats. "
                "The Growth plan supports up to 25 seats at $29 per seat per month. "
                "The Business plan supports up to 50 seats at $49 per seat per month. "
                "SSO is available on the Business and Enterprise plans only."
            ),
            metadata={"version": "1.0"},
        ),
        SourceDocument(
            doc_id="doc_test_003",
            title="Security Overview",
            source_type="guide",
            category="security",
            full_text=(
                "NovaSaaS supports SAML 2.0-based SSO with Okta, Azure AD, and Google Workspace. "
                "Data is encrypted at rest using AES-256. "
                "NovaSaaS holds SOC 2 Type II certification. "
                "SCIM provisioning is available on Business and Enterprise plans."
            ),
            metadata={"version": "1.0"},
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents) -> list[DocumentChunk]:
    """Pre-built chunks from sample documents."""
    chunks = []
    for doc in sample_documents:
        words = doc.full_text.split()
        mid = len(words) // 2
        for i, part_words in enumerate([words[:mid], words[mid:]]):
            text = " ".join(part_words)
            if len(part_words) >= 5:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}_chunk_{i:03d}",
                        doc_id=doc.doc_id,
                        chunk_text=text,
                        chunk_order=i,
                        word_count=len(part_words),
                        token_count_estimate=int(len(part_words) * 1.33),
                        metadata={"category": doc.category},
                    )
                )
    return chunks


@pytest.fixture
def sample_eval_examples() -> list[EvalExample]:
    return [
        EvalExample(
            example_id="test_eval_001",
            question="What is the refund window for NovaSaaS?",
            reference_answer="NovaSaaS offers a 30-day money-back guarantee.",
            expected_doc_ids=["doc_test_001"],
            expected_key_points=["30-day", "money-back"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
            is_answerable=True,
        ),
        EvalExample(
            example_id="test_eval_002",
            question="How many seats does the Business plan support?",
            reference_answer="The Business plan supports up to 50 seats at $49 per seat per month.",
            expected_doc_ids=["doc_test_002"],
            expected_key_points=["50 seats", "$49"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
            is_answerable=True,
        ),
        EvalExample(
            example_id="test_eval_003",
            question="What is the secret internal API key for NovaSaaS?",
            reference_answer="This information is not available in the documentation.",
            expected_doc_ids=[],
            expected_key_points=["not available"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.UNANSWERABLE,
            is_answerable=False,
        ),
    ]


@pytest.fixture
def sample_retrieved_chunks(sample_chunks) -> list[RetrievedChunk]:
    """Retrieved chunks with scores."""
    return [
        RetrievedChunk(
            chunk_id=sample_chunks[0].chunk_id,
            doc_id=sample_chunks[0].doc_id,
            chunk_text=sample_chunks[0].chunk_text,
            score=0.85,
            rank=1,
            metadata=sample_chunks[0].metadata,
        ),
        RetrievedChunk(
            chunk_id=sample_chunks[1].chunk_id,
            doc_id=sample_chunks[1].doc_id,
            chunk_text=sample_chunks[1].chunk_text,
            score=0.60,
            rank=2,
            metadata=sample_chunks[1].metadata,
        ),
    ]


@pytest.fixture
def tmp_dir() -> Path:
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
