"""LLM ports: a deterministic extractive answerer (tests/offline) and an
Anthropic-backed generator (production)."""

from __future__ import annotations

import os
import re
from typing import Protocol

from .prompt import build_prompt

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_NO_ANSWER = "I don't have enough information in the provided context to answer that."

# Common words carry no topical signal; counting them as overlap would let an
# off-topic question spuriously "match" a sentence instead of declining.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "from", "by", "with", "and", "or", "but",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "this", "that", "these", "those", "it", "its", "as", "about", "into",
    "i", "you", "he", "she", "they", "we", "me", "us", "them", "my", "your",
}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _content_tokens(text: str) -> set[str]:
    return _tokens(text) - _STOPWORDS


class LLMClient(Protocol):
    def answer(self, question: str, context: list[str]) -> str:
        ...


class ExtractiveLLM:
    """Deterministic, grounded answerer used in tests and offline demos.

    It returns the context sentence that best overlaps the question's terms, so
    the "answer" is always drawn verbatim from the retrieved context and can
    never hallucinate. A stand-in for a real generator with the same interface.
    """

    def answer(self, question: str, context: list[str]) -> str:
        if not context:
            return _NO_ANSWER
        q = _content_tokens(question)
        if not q:
            return _NO_ANSWER
        sentences = [s.strip() for chunk in context for s in _SENTENCE.split(chunk) if s.strip()]
        if not sentences:
            return _NO_ANSWER
        best = max(sentences, key=lambda s: len(q & _content_tokens(s)))
        if not (q & _content_tokens(best)):
            return _NO_ANSWER
        return best


class AnthropicLLM:
    """Generates answers with Claude. ``anthropic`` is imported lazily."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def answer(self, question: str, context: list[str]) -> str:  # pragma: no cover - needs API
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": build_prompt(question, context)}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
