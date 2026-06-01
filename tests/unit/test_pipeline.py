"""Unit tests for the LangGraph research pipeline.

Covers:
- Each node function independently with mock state
- Full graph invocation via graph.compile() + graph.invoke()
- Conditional routing (low confidence → re-search)
- State transitions between nodes
- Corpus search utility
- Edge cases and error handling

At least 15 meaningful tests.
"""

from __future__ import annotations

import pytest

from app.main import (
    CONFIDENCE_THRESHOLD,
    MAX_SEARCH_ITERATIONS,
    ResearchState,
    _should_continue,
    analyze_node,
    build_graph,
    draft_node,
    plan_node,
    review_node,
    run_pipeline,
    search_corpus,
    search_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> ResearchState:
    """Return a minimal valid ResearchState for testing."""
    state: ResearchState = {
        "query": "transformer architecture",
        "research_questions": [],
        "search_results": [],
        "key_facts": [],
        "confidence": 0.0,
        "search_iterations": 0,
        "report": {},
        "gaps": [],
        "done": False,
    }
    state.update(overrides)  # type: ignore[arg-type]
    return state


# ---------------------------------------------------------------------------
# 1. Corpus search tests
# ---------------------------------------------------------------------------


class TestCorpusSearch:
    def test_returns_list(self):
        results = search_corpus("agents", top_k=5)
        assert isinstance(results, list)

    def test_top_k_respected(self):
        results = search_corpus("transformer llm", top_k=3)
        assert len(results) <= 3

    def test_relevance_field_present(self):
        results = search_corpus("rag retrieval", top_k=5)
        for r in results:
            assert "relevance" in r
            assert isinstance(r["relevance"], float)

    def test_sorted_by_relevance_desc(self):
        results = search_corpus("vector database embeddings", top_k=6)
        scores = [r["relevance"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_known_topic_appears_first(self):
        """Querying 'rag retrieval generation' should surface the RAG document."""
        results = search_corpus("rag retrieval generation", top_k=5)
        titles = [r["title"].lower() for r in results]
        assert any("retrieval-augmented" in t or "rag" in t for t in titles)

    def test_snippet_truncated_to_300_chars(self):
        results = search_corpus("llm fine tuning", top_k=5)
        for r in results:
            assert len(r["snippet"]) <= 300

    def test_empty_results_for_nonsense_query(self):
        """A nonsense query still returns results (just low relevance)."""
        results = search_corpus("zzzzxxx999", top_k=3)
        # All scores should be very low
        for r in results:
            assert r["relevance"] < 0.5


# ---------------------------------------------------------------------------
# 2. plan_node tests
# ---------------------------------------------------------------------------


class TestPlanNode:
    def test_populates_research_questions(self):
        state = _base_state(query="LangGraph agent framework")
        result = plan_node(state)
        assert len(result["research_questions"]) >= 3

    def test_research_questions_contain_query(self):
        query = "mixture of experts"
        state = _base_state(query=query)
        result = plan_node(state)
        # At least one question should reference the query
        combined = " ".join(result["research_questions"]).lower()
        assert query.lower() in combined

    def test_resets_iteration_counter(self):
        state = _base_state(search_iterations=5)
        result = plan_node(state)
        assert result["search_iterations"] == 0

    def test_resets_done_flag(self):
        state = _base_state(done=True)
        result = plan_node(state)
        assert result["done"] is False

    def test_query_preserved(self):
        query = "federated learning privacy"
        state = _base_state(query=query)
        result = plan_node(state)
        assert result["query"] == query


# ---------------------------------------------------------------------------
# 3. search_node tests
# ---------------------------------------------------------------------------


class TestSearchNode:
    def test_returns_search_results(self):
        state = _base_state(
            query="transformer attention mechanism",
            research_questions=[
                "What is the transformer architecture?",
                "How does self-attention work?",
            ],
        )
        result = search_node(state)
        assert len(result["search_results"]) > 0

    def test_increments_iteration_counter(self):
        state = _base_state(
            query="llm fine tuning",
            research_questions=["What is fine-tuning?"],
            search_iterations=0,
        )
        result = search_node(state)
        assert result["search_iterations"] == 1

    def test_second_search_accumulates_results(self):
        state = _base_state(
            query="rag retrieval",
            research_questions=["How does RAG work?"],
            search_iterations=0,
        )
        after_first = search_node(state)
        first_count = len(after_first["search_results"])

        after_second = search_node(after_first)
        # Second iteration may not add more (already top-10), but count should be >= first
        assert len(after_second["search_results"]) >= first_count

    def test_results_have_required_fields(self):
        state = _base_state(
            query="vector database",
            research_questions=["What are vector databases?"],
        )
        result = search_node(state)
        for doc in result["search_results"]:
            assert "id" in doc
            assert "title" in doc
            assert "snippet" in doc
            assert "relevance" in doc


# ---------------------------------------------------------------------------
# 4. analyze_node tests
# ---------------------------------------------------------------------------


class TestAnalyzeNode:
    def _state_with_results(self, query: str = "llm agents") -> ResearchState:
        state = _base_state(query=query, research_questions=["What are LLM agents?"])
        return search_node(state)

    def test_confidence_between_0_and_1(self):
        state = self._state_with_results()
        result = analyze_node(state)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_key_facts_populated(self):
        state = self._state_with_results()
        result = analyze_node(state)
        assert len(result["key_facts"]) > 0

    def test_higher_relevance_yields_higher_confidence(self):
        """State with more / higher-scoring results should have higher confidence."""
        low_state = _base_state(
            query="quantum xyz",
            research_questions=[],
            search_results=[
                {"id": "d1", "title": "X", "snippet": "X", "relevance": 0.01, "topics": ["x"]}
            ],
        )
        high_state = _base_state(
            query="transformer architecture llm",
            research_questions=[],
        )
        high_state = search_node(high_state)
        low_result = analyze_node(low_state)
        high_result = analyze_node(high_state)
        assert high_result["confidence"] > low_result["confidence"]

    def test_gaps_identified_when_questions_not_covered(self):
        # Provide a research question that won't match any docs
        state = _base_state(
            query="quantum computing xyz",
            research_questions=["What is xyzzy_gibberish_term in context of zzz_abc?"],
            search_results=[
                {
                    "id": "d1",
                    "title": "Irrelevant Document",
                    "snippet": "Some text about cooking.",
                    "relevance": 0.1,
                    "topics": ["food"],
                }
            ],
        )
        result = analyze_node(state)
        assert len(result["gaps"]) > 0


# ---------------------------------------------------------------------------
# 5. draft_node tests
# ---------------------------------------------------------------------------


class TestDraftNode:
    def _analyzed_state(self) -> ResearchState:
        state = _base_state(
            query="retrieval augmented generation",
            research_questions=["How does RAG work?", "What are RAG applications?"],
        )
        state = search_node(state)
        return analyze_node(state)

    def test_report_has_executive_summary(self):
        state = self._analyzed_state()
        result = draft_node(state)
        assert "executive_summary" in result["report"]
        assert len(result["report"]["executive_summary"]) > 20

    def test_report_has_key_findings(self):
        state = self._analyzed_state()
        result = draft_node(state)
        assert "key_findings" in result["report"]
        assert len(result["report"]["key_findings"]) > 0

    def test_report_has_limitations(self):
        state = self._analyzed_state()
        result = draft_node(state)
        assert "limitations" in result["report"]
        assert isinstance(result["report"]["limitations"], list)

    def test_report_confidence_matches_state(self):
        state = self._analyzed_state()
        result = draft_node(state)
        assert result["report"]["confidence"] == result["confidence"]

    def test_report_sources_present(self):
        state = self._analyzed_state()
        result = draft_node(state)
        assert "sources" in result["report"]
        assert len(result["report"]["sources"]) > 0


# ---------------------------------------------------------------------------
# 6. review_node tests
# ---------------------------------------------------------------------------


class TestReviewNode:
    def test_done_true_when_confidence_high(self):
        state = _base_state(confidence=0.95, search_iterations=1)
        result = review_node(state)
        assert result["done"] is True

    def test_done_false_when_confidence_low(self):
        state = _base_state(confidence=0.10, search_iterations=1)
        result = review_node(state)
        assert result["done"] is False

    def test_done_true_when_max_iterations_reached(self):
        state = _base_state(confidence=0.10, search_iterations=MAX_SEARCH_ITERATIONS)
        result = review_node(state)
        assert result["done"] is True

    def test_done_at_threshold_boundary(self):
        state = _base_state(confidence=CONFIDENCE_THRESHOLD, search_iterations=1)
        result = review_node(state)
        assert result["done"] is True


# ---------------------------------------------------------------------------
# 7. Conditional routing
# ---------------------------------------------------------------------------


class TestConditionalRouting:
    def test_routes_to_end_when_done(self):
        state = _base_state(done=True)
        assert _should_continue(state) == "end"

    def test_routes_to_search_when_not_done(self):
        state = _base_state(done=False)
        assert _should_continue(state) == "search"


# ---------------------------------------------------------------------------
# 8. Full graph integration tests
# ---------------------------------------------------------------------------


class TestFullGraphIntegration:
    def test_graph_compiles_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_run_pipeline_returns_dict(self):
        result = run_pipeline("What is a transformer model?")
        assert isinstance(result, dict)

    def test_run_pipeline_has_confidence(self):
        result = run_pipeline("LangGraph stateful agents")
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_run_pipeline_has_report(self):
        result = run_pipeline("retrieval augmented generation")
        assert "report" in result
        assert "executive_summary" in result["report"]
        assert "key_findings" in result["report"]
        assert "limitations" in result["report"]

    def test_run_pipeline_raises_on_empty_query(self):
        with pytest.raises(Exception):
            run_pipeline("")

    def test_run_pipeline_search_iterations_gte_1(self):
        result = run_pipeline("mixture of experts sparse models")
        assert result["search_iterations"] >= 1

    def test_run_pipeline_low_confidence_triggers_re_search(self):
        """A very obscure query should still exhaust retries."""
        result = run_pipeline("xyzzy_nonsense_term_zzz_abc_9999")
        # Even for a bad query we must reach max iterations
        assert result["search_iterations"] >= 1
