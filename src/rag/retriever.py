"""Indexes documents (chunk -> embed -> store) and retrieves relevant chunks."""

from __future__ import annotations

from .chunking import chunk_text
from .embeddings import Embedder
from .vector_store import Document, InMemoryVectorStore, Scored


class Retriever:
    def __init__(self, embedder: Embedder, chunk_words: int = 120, overlap: int = 30) -> None:
        self._store = InMemoryVectorStore(embedder=embedder)
        self._chunk_words = chunk_words
        self._overlap = overlap

    def add_document(self, source: str, text: str) -> int:
        chunks = chunk_text(text, self._chunk_words, self._overlap)
        for i, chunk in enumerate(chunks):
            self._store.add(Document(id=f"{source}#{i}", text=chunk, source=source))
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> list[Scored]:
        return self._store.search(query, k)

    def __len__(self) -> int:
        return len(self._store)
