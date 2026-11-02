"""The RAG pipeline: retrieve relevant chunks, then answer grounded in them."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient
from .retriever import Retriever


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]
    contexts: list[str]


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, k: int = 3) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def answer(self, question: str) -> AnswerResult:
        hits = self.retriever.retrieve(question, self.k)
        contexts = [h.document.text for h in hits]
        answer = self.llm.answer(question, contexts)

        sources: list[str] = []
        for hit in hits:
            if hit.document.source not in sources:
                sources.append(hit.document.source)

        return AnswerResult(answer=answer, sources=sources, contexts=contexts)
