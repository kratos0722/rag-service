"""Split documents into overlapping, retrieval-sized chunks."""

from __future__ import annotations

import re

_WORD = re.compile(r"\S+")


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 30) -> list[str]:
    """Split ``text`` into chunks of ~``chunk_words`` words with ``overlap``.

    Overlap keeps context that straddles a boundary retrievable from either side.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be in [0, chunk_words)")

    words = _WORD.findall(text)
    if not words:
        return []

    step = chunk_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_words]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks
