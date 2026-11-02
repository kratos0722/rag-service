"""Automated output evals for the RAG pipeline.

Scores answers on three axes a RAG system must get right: relevance (did it
contain the expected facts?), groundedness (is the answer supported by the
retrieved context rather than invented?), and source attribution (did it cite
the right document?).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pipeline import RagPipeline

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]
    expected_source: str | None = None


@dataclass
class EvalReport:
    total: int
    relevant: int
    grounded: int
    source_correct: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def _is_grounded(answer: str, contexts: list[str], threshold: float = 0.8) -> bool:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return False
    context_tokens = _tokens(" ".join(contexts))
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return overlap >= threshold


def evaluate(pipeline: RagPipeline, cases: list[EvalCase]) -> EvalReport:
    relevant = grounded = source_correct = passed = 0
    for case in cases:
        result = pipeline.answer(case.question)
        answer_lower = result.answer.lower()

        is_relevant = all(kw.lower() in answer_lower for kw in case.expected_keywords)
        is_grounded = _is_grounded(result.answer, result.contexts)
        is_source_ok = case.expected_source is None or case.expected_source in result.sources

        relevant += is_relevant
        grounded += is_grounded
        source_correct += is_source_ok
        if is_relevant and is_grounded and is_source_ok:
            passed += 1

    return EvalReport(
        total=len(cases),
        relevant=relevant,
        grounded=grounded,
        source_correct=source_correct,
        passed=passed,
    )
