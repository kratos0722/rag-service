"""Alert triage grounded in runbook history.

This is a focused application of the RAG pipeline: instead of answering an
arbitrary question, it reads an incoming alert, retrieves the most relevant
runbook passages from the same vector index, and returns grounded next steps
plus a routing decision (priority + where to send it).

The retrieval + grounded-answer machinery is exactly the RAG core (``Retriever``
+ ``LLMClient``); this module only adds the alert-specific match gate and the
priority/routing policy on top, so there is no second copy of the RAG logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pipeline import RagPipeline

# Severities we accept, mapped to a base priority (P1 = most urgent).
_SEVERITY_TO_PRIORITY = {"critical": 1, "high": 2, "medium": 3, "low": 4}

_ROUTE_BY_PRIORITY = {
    1: "page on-call immediately",
    2: "page on-call",
    3: "create ticket (next business day)",
    4: "log for review",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "at", "for", "from", "by", "with", "and", "or", "it", "its", "this", "that",
    "has", "have", "not", "no", "we", "i",
}

_NO_RUNBOOK = (
    "No matching runbook found. Escalate to on-call, work the incident from first "
    "principles, and capture a new runbook once it is resolved."
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


@dataclass
class Alert:
    """An incoming alert to triage."""

    message: str
    service: str | None = None
    severity: str = "medium"  # critical | high | medium | low


@dataclass
class TriageResult:
    matched: bool
    priority: str  # "P1".."P4"
    route: str
    suggested_steps: str
    matched_runbooks: list[str]
    contexts: list[str]
    confidence: float  # retrieval score of the best hit, [0, 1]


class TriageService:
    """Triages alerts against an index of runbooks using the RAG pipeline."""

    def __init__(self, pipeline: RagPipeline, match_threshold: float = 0.05) -> None:
        self._pipeline = pipeline
        self._match_threshold = match_threshold

    def ingest_runbook(self, source: str, text: str) -> int:
        """Index a runbook so future alerts can be matched against it."""
        return self._pipeline.retriever.add_document(source, text)

    @staticmethod
    def _query(alert: Alert) -> str:
        parts = [alert.message]
        if alert.service:
            parts.append(f"service {alert.service}")
        return " ".join(parts)

    def triage(self, alert: Alert) -> TriageResult:
        if alert.severity not in _SEVERITY_TO_PRIORITY:
            raise ValueError(f"unknown severity: {alert.severity!r}")

        query = self._query(alert)
        hits = self._pipeline.retriever.retrieve(query, self._pipeline.k)

        # Match gate: a runbook must both rank above the score threshold AND
        # share a meaningful term with the alert. The token check keeps the
        # decision robust even when the embedder assigns small non-zero cosine
        # similarity to unrelated text.
        best = hits[0] if hits else None
        query_terms = _content_tokens(query)
        overlap = query_terms & _content_tokens(best.document.text) if best else set()
        matched = bool(best and best.score >= self._match_threshold and overlap)

        confidence = round(best.score, 4) if best else 0.0

        if matched:
            contexts = [h.document.text for h in hits]
            suggested_steps = self._pipeline.llm.answer(query, contexts)
            sources: list[str] = []
            for hit in hits:
                if hit.document.source not in sources:
                    sources.append(hit.document.source)
        else:
            contexts = []
            suggested_steps = _NO_RUNBOOK
            sources = []

        priority_num = self._priority(alert.severity, matched)
        route = self._route(priority_num, matched, sources[0] if sources else None)

        return TriageResult(
            matched=matched,
            priority=f"P{priority_num}",
            route=route,
            suggested_steps=suggested_steps,
            matched_runbooks=sources,
            contexts=contexts,
            confidence=confidence,
        )

    @staticmethod
    def _priority(severity: str, matched: bool) -> int:
        base = _SEVERITY_TO_PRIORITY[severity]
        # An unknown failure with no runbook is riskier than a known one, so
        # escalate it one level (never past P1, and never escalate "low").
        if not matched and severity != "low":
            return max(1, base - 1)
        return base

    @staticmethod
    def _route(priority_num: int, matched: bool, primary_source: str | None) -> str:
        base = _ROUTE_BY_PRIORITY[priority_num]
        if matched and primary_source:
            return f"{base}; follow runbook: {primary_source}"
        return f"{base}; no matching runbook — escalate to a human"
