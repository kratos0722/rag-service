"""FastAPI app exposing ingest + ask over a process-local RAG index."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .factory import build_pipeline
from .triage import Alert, TriageService

pipeline = build_pipeline()
# The triage service shares the same retriever/index, so runbooks ingested via
# /api/ingest are immediately available for alert triage.
triage_service = TriageService(pipeline)
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


class TriageRequest(BaseModel):
    message: str = Field(min_length=1)
    service: str | None = None
    severity: str = Field(default="medium")


class TriageResponse(BaseModel):
    matched: bool
    priority: str
    route: str
    suggested_steps: str
    matched_runbooks: list[str]
    confidence: float


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


@app.post("/api/triage", response_model=TriageResponse)
def triage(body: TriageRequest) -> TriageResponse:
    result = triage_service.triage(
        Alert(message=body.message, service=body.service, severity=body.severity)
    )
    return TriageResponse(
        matched=result.matched,
        priority=result.priority,
        route=result.route,
        suggested_steps=result.suggested_steps,
        matched_runbooks=result.matched_runbooks,
        confidence=result.confidence,
    )
