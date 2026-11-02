from rag.embeddings import HashingEmbedder
from rag.vector_store import Document, InMemoryVectorStore


def _store():
    store = InMemoryVectorStore(embedder=HashingEmbedder())
    store.add(Document("d1", "cats and dogs are common household pets", "pets.md"))
    store.add(Document("d2", "the stock market fell sharply today", "finance.md"))
    store.add(Document("d3", "dogs need daily walks and training", "pets.md"))
    return store


def test_search_returns_most_relevant_first():
    results = _store().search("tell me about dogs", k=2)
    assert len(results) == 2
    assert results[0].document.id in {"d1", "d3"}
    assert results[0].score >= results[1].score


def test_search_empty_store():
    empty = InMemoryVectorStore(embedder=HashingEmbedder())
    assert empty.search("anything") == []
