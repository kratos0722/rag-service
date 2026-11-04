"""Tests for alert triage grounded in runbook history."""

from __future__ import annotations

import pytest

from rag.factory import build_triage_service
from rag.triage import Alert, TriageService


RUNBOOKS = {
    "disk_full.md": (
        "Runbook: Disk Space Full. Symptoms: disk usage near 100 percent, failed "
        "writes, log rotation errors. Steps: find the largest directories, rotate "
        "and compress old logs, expand the attached volume, check for runaway log output."
    ),
    "high_cpu.md": (
        "Runbook: High CPU Utilization. Symptoms: sustained CPU above 90 percent and "
        "rising latency. Steps: identify the hot process, scale out the service, add "
        "caching for the hot endpoint, roll back the most recent deploy."
    ),
    "db_connections.md": (
        "Runbook: Database connection pool exhausted. Symptoms: too many connections "
        "errors and timeouts. Steps: kill idle sessions, increase the pool size, add "
        "a pooler such as PgBouncer."
    ),
}


def _service() -> TriageService:
    svc = build_triage_service(k=3)
    for source, text in RUNBOOKS.items():
        svc.ingest_runbook(source, text)
    return svc


def test_matching_alert_is_grounded_and_prioritized():
    svc = _service()
    result = svc.triage(Alert(message="disk usage is full, writes are failing", service="web-01", severity="critical"))
    assert result.matched is True
    assert "disk_full.md" in result.matched_runbooks
    # Grounded: the suggested steps come from the matched runbook text.
    assert "disk" in result.suggested_steps.lower()
    assert result.priority == "P1"  # critical
    assert "follow runbook" in result.route


def test_severity_maps_to_priority_for_known_incident():
    svc = _service()
    result = svc.triage(Alert(message="database connection pool exhausted, too many connections", severity="medium"))
    assert result.matched is True
    assert "db_connections.md" in result.matched_runbooks
    assert result.priority == "P3"  # medium, known runbook -> no escalation


def test_novel_low_severity_alert_is_not_matched():
    svc = _service()
    result = svc.triage(Alert(message="flux capacitor desynchronized warp coil", severity="low"))
    assert result.matched is False
    assert result.matched_runbooks == []
    assert result.priority == "P4"
    assert "no matching runbook" in result.route
    assert result.suggested_steps.startswith("No matching runbook")


def test_novel_high_severity_alert_is_escalated():
    svc = _service()
    # Unknown failure + high severity -> escalate one level (P2 -> P1).
    result = svc.triage(Alert(message="flux capacitor desynchronized warp coil", severity="high"))
    assert result.matched is False
    assert result.priority == "P1"
    assert "escalate to a human" in result.route


def test_unknown_severity_rejected():
    svc = _service()
    with pytest.raises(ValueError):
        svc.triage(Alert(message="disk full", severity="sev0"))


def test_confidence_reflects_retrieval_score():
    svc = _service()
    matched = svc.triage(Alert(message="high cpu utilization, process saturating the host", severity="high"))
    assert matched.matched is True
    assert 0.0 < matched.confidence <= 1.0
