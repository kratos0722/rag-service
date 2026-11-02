"""In-memory vector store with cosine top-k search."""

from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings import Embedder, cosine


@dataclass
class Document:
    id: str
    text: str
    source: str


@dataclass
class Scored:
    document: Document
    score: float


@dataclass
class InMemoryVectorStore:
    """A tiny vector index. Production would use FAISS / pgvector; the interface
    (add + search) is the same."""

    embedder: Embedder
    _docs: list[Document] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)

    def add(self, document: Document) -> None:
        self._docs.append(document)
        self._vectors.append(self.embedder.embed(document.text))

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, k: int = 3) -> list[Scored]:
        if not self._docs:
            return []
        qv = self.embedder.embed(query)
        scored = [Scored(doc, cosine(qv, vec)) for doc, vec in zip(self._docs, self._vectors)]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]
