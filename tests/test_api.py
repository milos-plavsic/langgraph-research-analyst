"""API-level tests for the LangGraph Research Analyst service."""

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    """Health endpoint returns 200 and status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_research_pipeline_returns_200() -> None:
    """POST /v1/research returns 200 for a valid query."""
    r = client.post("/v1/research", json={"query": "Compare agent frameworks"})
    assert r.status_code == 200


def test_research_pipeline_response_shape() -> None:
    """Response contains required top-level keys."""
    r = client.post("/v1/research", json={"query": "transformer architecture"})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "transformer architecture"
    assert "report" in data
    assert data["confidence"] > 0


def test_research_report_keys() -> None:
    """The nested report dict contains expected sub-keys."""
    r = client.post("/v1/research", json={"query": "retrieval augmented generation"})
    assert r.status_code == 200
    report = r.json()["report"]
    assert "executive_summary" in report
    assert "key_findings" in report
    assert "limitations" in report


def test_finetune_playbook() -> None:
    """GET /v1/finetune/playbook returns 200 with a stack field."""
    r = client.get("/v1/finetune/playbook")
    assert r.status_code == 200
    assert "stack" in r.json()
