"""FastAPI app exposing ingest + ask over a process-local RAG index."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .factory import build_pipeline

pipeline = build_pipeline()
app = FastAPI(title="rag-service", version="0.1.0")


class IngestRequest(BaseModel):
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    source: str
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP"}


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest) -> IngestResponse:
    chunks = pipeline.retriever.add_document(body.source, body.text)
    return IngestResponse(source=body.source, chunks=chunks)


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    result = pipeline.answer(body.question)
    return AskResponse(answer=result.answer, sources=result.sources)
