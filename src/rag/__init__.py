"""Retrieval-augmented generation (RAG) over a document corpus.

Documents are chunked, embedded, and stored in a vector index; a question is
embedded, the most similar chunks are retrieved, and an LLM answers *grounded in
those chunks* with citations. The embedder and the LLM are both ports, so the
whole pipeline runs deterministically offline in tests (hashing embedder +
extractive answerer) and swaps in real models in production.
"""

__version__ = "0.1.0"
