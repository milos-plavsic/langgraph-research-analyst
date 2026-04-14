from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_research_pipeline() -> None:
    r = client.post("/v1/research", json={"query": "Compare agent frameworks"})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "Compare agent frameworks"
    assert "draft" in data
    assert data["confidence"] > 0
