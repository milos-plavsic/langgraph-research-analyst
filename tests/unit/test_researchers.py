"""Unit tests for corpus and web researcher nodes."""

from __future__ import annotations

from unittest.mock import patch

from app.main import ResearchState
from app.researchers import corpus_researcher_node, web_researcher_node


def _state(**overrides) -> ResearchState:
    base: ResearchState = {
        "query": "transformer architecture",
        "research_questions": ["What is a transformer?"],
        "search_results": [],
        "key_facts": [],
        "confidence": 0.0,
        "search_iterations": 0,
        "report": {},
        "gaps": [],
        "done": False,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_corpus_researcher_adds_documents() -> None:
    out = corpus_researcher_node(_state())
    assert len(out["search_results"]) > 0
    assert out["search_results"][0].get("researcher") == "corpus"


def test_web_researcher_adds_wikipedia_hit() -> None:
    fake = {"title": "Transformer", "content": "Attention-based model architecture."}
    with patch("agent_core.fetch_wikipedia_summary", return_value=fake):
        out = web_researcher_node(_state())
    assert any(r.get("researcher") == "web" for r in out["search_results"])
