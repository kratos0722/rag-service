import pytest

from rag.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", chunk_words=120) == ["hello world"]


def test_chunks_overlap():
    words = " ".join(str(i) for i in range(300))
    chunks = chunk_text(words, chunk_words=100, overlap=20)
    assert len(chunks) >= 3
    # Each chunk is at most chunk_words long.
    assert all(len(c.split()) <= 100 for c in chunks)
    # Overlap: the tail of chunk 0 reappears at the head of chunk 1.
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=5, overlap=5)
