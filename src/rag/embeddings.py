"""Embedding ports and a dependency-free, deterministic implementation."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class HashingEmbedder:
    """A deterministic bag-of-words hashing embedder.

    No model download, no randomness (unlike Python's salted ``hash``): tokens
    are hashed into a fixed number of buckets and the vector is L2-normalized, so
    cosine similarity reflects shared vocabulary. Good enough to exercise real
    retrieval behavior in tests; swap in a transformer embedder for production.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            vec[_stable_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs from HashingEmbedder are already normalized."""
    return sum(x * y for x, y in zip(a, b))
