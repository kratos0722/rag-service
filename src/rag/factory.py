"""Builds a default pipeline: hashing embedder + (Anthropic if a key is set,
else the offline extractive answerer)."""

from __future__ import annotations

import os

from .embeddings import HashingEmbedder
from .llm import AnthropicLLM, ExtractiveLLM, LLMClient
from .pipeline import RagPipeline
from .retriever import Retriever


def build_pipeline(k: int = 3) -> RagPipeline:
    retriever = Retriever(HashingEmbedder())
    llm: LLMClient = AnthropicLLM() if os.getenv("ANTHROPIC_API_KEY") else ExtractiveLLM()
    return RagPipeline(retriever, llm, k=k)
