import math

from rag.embeddings import HashingEmbedder, cosine


def test_embedding_is_deterministic():
    emb = HashingEmbedder()
    assert emb.embed("the quick brown fox") == emb.embed("the quick brown fox")


def test_embedding_is_normalized():
    vec = HashingEmbedder().embed("some non empty text")
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)


def test_similar_text_scores_higher_than_dissimilar():
    emb = HashingEmbedder()
    base = emb.embed("the cat sat on the mat")
    similar = emb.embed("a cat on a mat")
    dissimilar = emb.embed("quantum chromodynamics lecture notes")
    assert cosine(base, similar) > cosine(base, dissimilar)


def test_empty_text_is_zero_vector():
    assert all(v == 0.0 for v in HashingEmbedder().embed(""))
