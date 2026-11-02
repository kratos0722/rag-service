"""Grounded prompt construction for the generative LLM adapter."""

from __future__ import annotations


def build_prompt(question: str, context: list[str]) -> str:
    """Build a prompt that constrains the model to the retrieved context."""
    numbered = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context))
    return (
        "You are a careful assistant. Answer the question using ONLY the context "
        "below. If the context does not contain the answer, say you don't know. "
        "Cite the sources you used as bracketed numbers, e.g. [1].\n\n"
        f"Context:\n{numbered}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
