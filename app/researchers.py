"""Dedicated researcher nodes for corpus and web evidence collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_core import configure_logging

if TYPE_CHECKING:
    from app.main import ResearchState

logger = configure_logging("researchers")


def corpus_researcher_node(state: ResearchState) -> ResearchState:
    """Retrieve evidence from the in-memory technical corpus."""
    from app.main import ResearchState, search_corpus

    query = state["query"]
    questions = state.get("research_questions", [])
    seen = {r["id"] for r in state.get("search_results", [])}
    collected: list[dict] = []

    for q in [query, *questions[:2]]:
        for doc in search_corpus(q, top_k=3):
            if doc["id"] not in seen:
                seen.add(doc["id"])
                collected.append({**doc, "researcher": "corpus"})

    logger.info("corpus researcher added %s documents", len(collected))
    merged = list(state.get("search_results", [])) + collected
    return ResearchState(**{**state, "search_results": merged})


def web_researcher_node(state: ResearchState) -> ResearchState:
    """Retrieve live web summaries to complement static corpus hits."""
    from app.main import ResearchState, _tokenise

    query = state["query"]
    seen = {r["id"] for r in state.get("search_results", [])}
    collected: list[dict] = []

    try:
        from agent_core import fetch_wikipedia_summary

        wiki = fetch_wikipedia_summary(query)
        if wiki.get("content"):
            wiki_id = f"wiki_{abs(hash(query)) % 100_000}"
            if wiki_id not in seen:
                collected.append(
                    {
                        "id": wiki_id,
                        "title": str(wiki.get("title", query)),
                        "snippet": str(wiki["content"])[:500],
                        "relevance": 0.85,
                        "topics": _tokenise(query)[:6],
                        "researcher": "web",
                    }
                )
    except Exception as exc:
        logger.warning("web researcher skipped: %s", exc)

    merged = list(state.get("search_results", [])) + collected
    return ResearchState(**{**state, "search_results": merged})
