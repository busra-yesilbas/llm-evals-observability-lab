"""
LLM Evals & Observability Lab
==============================
A production-grade evaluation and observability framework for RAG and
agentic LLM systems. Designed for rigorous measurement, experiment
tracking, trace inspection, and failure analysis.

Quick start
-----------
>>> from llm_evals_lab.config import load_config
>>> from llm_evals_lab.generation.rag_pipeline import RAGPipeline
>>> cfg = load_config()
>>> pipeline = RAGPipeline.from_config(cfg)
>>> result = pipeline.run("What is the refund policy?")
"""

__version__ = "0.1.0"
__author__ = "LLM Evals Lab"
