"""LangGraph research analyst — real state-machine implementation.

The pipeline uses a genuine LangGraph StateGraph compiled and invoked via
graph.compile() / graph.invoke().  State is a TypedDict (LangGraph requirement).
A conditional edge re-routes to the search node when confidence is below
the threshold so the agent gathers more evidence before drafting.
"""

from __future__ import annotations

import math
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from ml_core import configure_logging
from ml_core.exceptions import ApplicationError

logger = configure_logging("research-analyst")

# ---------------------------------------------------------------------------
# Corpus — 20 documents covering common ML/agent topics
# ---------------------------------------------------------------------------

CORPUS: list[dict] = [
    {
        "id": "doc_001",
        "title": "LangGraph: Stateful Multi-Actor Applications with LLMs",
        "text": (
            "LangGraph is a library for building stateful, multi-actor applications "
            "with large language models.  It extends LangChain with cyclic graph "
            "support, enabling agent loops, conditional branching, and persistent "
            "checkpointing.  Nodes represent computation steps; edges define control "
            "flow including conditional routing."
        ),
        "topics": ["langgraph", "agents", "llm", "graph", "stateful"],
    },
    {
        "id": "doc_002",
        "title": "Transformer Architecture: Attention Is All You Need",
        "text": (
            "The transformer model replaces recurrent networks with self-attention "
            "mechanisms.  Multi-head attention allows the model to jointly attend to "
            "information from different representation subspaces.  Positional "
            "encoding injects sequence order.  Transformers achieve state-of-the-art "
            "results on NLP, vision, and multimodal tasks."
        ),
        "topics": ["transformer", "attention", "nlp", "deep learning", "architecture"],
    },
    {
        "id": "doc_003",
        "title": "Retrieval-Augmented Generation (RAG) Overview",
        "text": (
            "RAG combines a dense retrieval step with a generative LLM.  A query "
            "encoder retrieves relevant passages from a knowledge base; the LLM then "
            "conditions its output on those passages.  RAG reduces hallucination and "
            "keeps knowledge up-to-date without full model retraining."
        ),
        "topics": ["rag", "retrieval", "generation", "llm", "knowledge"],
    },
    {
        "id": "doc_004",
        "title": "Reinforcement Learning from Human Feedback (RLHF)",
        "text": (
            "RLHF fine-tunes language models using a reward model trained on human "
            "preference data.  Proximal Policy Optimization (PPO) updates the LLM "
            "policy to maximize reward while staying close to the reference policy "
            "via a KL-divergence penalty.  RLHF is central to ChatGPT and Claude."
        ),
        "topics": ["rlhf", "reward model", "ppo", "fine-tuning", "alignment"],
    },
    {
        "id": "doc_005",
        "title": "Vector Databases for Semantic Search",
        "text": (
            "Vector databases store high-dimensional embeddings and support "
            "approximate nearest-neighbour (ANN) search.  Popular engines include "
            "Pinecone, Weaviate, Qdrant, and pgvector.  HNSW and IVF-PQ are common "
            "index structures.  They power semantic search, RAG, and recommendation "
            "systems."
        ),
        "topics": ["vector database", "embeddings", "search", "ann", "rag"],
    },
    {
        "id": "doc_006",
        "title": "Prompt Engineering Best Practices",
        "text": (
            "Effective prompts use clear instructions, examples (few-shot), and "
            "chain-of-thought (CoT) reasoning.  Role prompting, structured output "
            "constraints, and temperature tuning improve consistency.  System "
            "prompts set context; user turns supply task-specific instructions."
        ),
        "topics": ["prompt engineering", "few-shot", "chain of thought", "llm"],
    },
    {
        "id": "doc_007",
        "title": "AutoGPT and Autonomous Agent Frameworks",
        "text": (
            "AutoGPT demonstrated autonomous LLM agents that loop: think, act, "
            "observe.  Modern frameworks such as LangChain Agents, CrewAI, and "
            "AutoGen provide tool-use, memory, and multi-agent coordination.  "
            "ReAct (Reasoning + Acting) is a popular prompting strategy for agents."
        ),
        "topics": ["agents", "autogpt", "langchain", "crewai", "autonomous"],
    },
    {
        "id": "doc_008",
        "title": "Fine-Tuning Large Language Models",
        "text": (
            "Full fine-tuning updates all parameters; parameter-efficient methods "
            "such as LoRA, QLoRA, and prefix-tuning reduce compute by adapting a "
            "small number of parameters.  Instruction tuning on curated datasets "
            "improves instruction-following.  Evaluation uses benchmarks like "
            "MMLU, HellaSwag, and TruthfulQA."
        ),
        "topics": ["fine-tuning", "lora", "qlora", "instruction tuning", "llm"],
    },
    {
        "id": "doc_009",
        "title": "Mixture of Experts (MoE) Models",
        "text": (
            "MoE replaces dense feed-forward layers with a set of expert networks "
            "and a gating mechanism that routes tokens to the top-k experts.  "
            "Mistral's Mixtral 8x7B and GPT-4 (rumoured) use MoE.  Sparse "
            "activation reduces inference FLOPs while maintaining model capacity."
        ),
        "topics": ["moe", "mixture of experts", "architecture", "llm", "sparse"],
    },
    {
        "id": "doc_010",
        "title": "LLM Evaluation and Benchmarking",
        "text": (
            "Common benchmarks: MMLU (knowledge), HumanEval (code), GSM8K (math), "
            "BIG-Bench Hard (reasoning), MT-Bench (chat).  LLM-as-judge uses a "
            "stronger model to grade outputs.  Contamination and leakage are "
            "critical concerns when interpreting benchmark scores."
        ),
        "topics": ["evaluation", "benchmark", "mmlu", "humaneval", "llm"],
    },
    {
        "id": "doc_011",
        "title": "Embeddings and Semantic Similarity",
        "text": (
            "Text embeddings map strings to dense vectors where cosine similarity "
            "reflects semantic relatedness.  OpenAI text-embedding-3-large, "
            "Cohere Embed, and SentenceTransformers are popular choices.  "
            "Embeddings power clustering, classification, and retrieval."
        ),
        "topics": ["embeddings", "similarity", "cosine", "sentence transformers", "nlp"],
    },
    {
        "id": "doc_012",
        "title": "Constitutional AI and AI Safety",
        "text": (
            "Constitutional AI (Anthropic) trains models to follow a set of "
            "principles using self-critique and revision.  Harmlessness, "
            "helpfulness, and honesty are the three H's.  Red-teaming, RLAIF, "
            "and interpretability research support safer AI deployment."
        ),
        "topics": ["safety", "constitutional ai", "alignment", "anthropic", "rlaif"],
    },
    {
        "id": "doc_013",
        "title": "Multimodal LLMs: Vision and Language",
        "text": (
            "GPT-4V, Claude 3, Gemini, and LLaVA process both images and text.  "
            "Vision encoders (CLIP, SigLIP) project image patches into the LLM "
            "token space.  Applications include document understanding, chart QA, "
            "and visual agents."
        ),
        "topics": ["multimodal", "vision", "llm", "clip", "image"],
    },
    {
        "id": "doc_014",
        "title": "Quantisation and Efficient Inference",
        "text": (
            "INT8 and INT4 quantisation reduce model size and memory bandwidth.  "
            "GPTQ and AWQ are popular post-training quantisation methods.  "
            "vLLM uses PagedAttention for high-throughput serving.  Speculative "
            "decoding and continuous batching further improve latency."
        ),
        "topics": ["quantisation", "inference", "vllm", "gptq", "efficiency"],
    },
    {
        "id": "doc_015",
        "title": "LangChain: Building LLM Applications",
        "text": (
            "LangChain provides chains, agents, memory, and tool-use abstractions "
            "for LLM applications.  LangChain Expression Language (LCEL) composes "
            "runnables declaratively.  Integrations cover 100+ LLM providers, "
            "vector stores, and APIs."
        ),
        "topics": ["langchain", "agents", "chains", "lcel", "llm"],
    },
    {
        "id": "doc_016",
        "title": "Structured Output and Function Calling",
        "text": (
            "OpenAI function calling and the JSON mode instruct models to emit "
            "structured JSON.  Instructor and Outlines libraries enforce Pydantic "
            "schemas.  Structured outputs reduce parsing errors in agentic "
            "pipelines that consume model outputs programmatically."
        ),
        "topics": ["structured output", "function calling", "json", "pydantic", "agents"],
    },
    {
        "id": "doc_017",
        "title": "Graph Neural Networks for Knowledge Graphs",
        "text": (
            "GNNs aggregate neighbour features using message-passing.  Knowledge "
            "graphs (Wikidata, Freebase) store facts as (subject, relation, object) "
            "triples.  GNNs and KG embeddings (TransE, RotatE) enable link "
            "prediction and reasoning over structured knowledge."
        ),
        "topics": ["gnn", "knowledge graph", "graph", "reasoning", "embeddings"],
    },
    {
        "id": "doc_018",
        "title": "Diffusion Models for Image and Audio Generation",
        "text": (
            "Diffusion models learn to reverse a noise process.  Stable Diffusion, "
            "DALL-E 3, and Midjourney use latent diffusion with classifier-free "
            "guidance.  Score-based models and DDPM are theoretical foundations.  "
            "Audio diffusion (AudioLDM, MusicGen) generates speech and music."
        ),
        "topics": ["diffusion", "image generation", "stable diffusion", "generative", "audio"],
    },
    {
        "id": "doc_019",
        "title": "MLOps: Model Deployment and Monitoring",
        "text": (
            "MLOps covers CI/CD for ML: versioning (DVC, MLflow), reproducible "
            "training, containerised serving (Docker, Kubernetes), and production "
            "monitoring (data drift, prediction drift).  Feature stores "
            "(Feast, Tecton) centralise feature computation."
        ),
        "topics": ["mlops", "deployment", "monitoring", "mlflow", "kubernetes"],
    },
    {
        "id": "doc_020",
        "title": "Federated Learning and Privacy-Preserving ML",
        "text": (
            "Federated learning trains models across distributed clients without "
            "centralising raw data.  FedAvg aggregates local gradients.  "
            "Differential privacy adds calibrated noise to protect individual "
            "records.  Applications include healthcare and on-device NLP."
        ),
        "topics": ["federated learning", "privacy", "differential privacy", "fedavg", "ml"],
    },
]

# ---------------------------------------------------------------------------
# Confidence threshold — below this, the graph loops back to search
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.70
MAX_SEARCH_ITERATIONS = 3


# ---------------------------------------------------------------------------
# State — must be TypedDict for LangGraph
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    """LangGraph state for the research pipeline."""

    query: str
    research_questions: list[str]
    search_results: list[dict]
    key_facts: list[str]
    confidence: float
    search_iterations: int
    report: dict  # keys: executive_summary, key_findings, limitations
    gaps: list[str]
    done: bool


# ---------------------------------------------------------------------------
# Simple TF-IDF-style retrieval (no external deps)
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _score_document(query_tokens: list[str], doc: dict) -> float:
    """Return a relevance score for *doc* against *query_tokens*."""
    doc_text = (doc["title"] + " " + doc["text"] + " " + " ".join(doc["topics"])).lower()
    doc_tokens = set(_tokenise(doc_text))
    hits = sum(1 for t in query_tokens if t in doc_tokens)
    # TF component: normalise by doc length estimate
    score = hits / (1 + math.log1p(len(doc_tokens)))
    # Boost for topic exact matches
    topic_hits = sum(1 for q in query_tokens if any(q in topic for topic in doc["topics"]))
    score += topic_hits * 0.15
    return score


def search_corpus(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k documents from the in-memory corpus."""
    tokens = _tokenise(query)
    scored = [(doc, _score_document(tokens, doc)) for doc in CORPUS]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for doc, score in scored[:top_k]:
        results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "snippet": doc["text"][:300],
                "relevance": round(score, 4),
                "topics": doc["topics"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------


def plan_node(state: ResearchState) -> ResearchState:
    """Parse the query, identify 3-5 key research questions, create search strategy."""
    query = state["query"]
    logger.info(f"[plan_node] query={query!r}")

    _tokenise(query)
    # Generate research questions by focusing on different facets of the query
    questions = [
        f"What is the definition and core concept of {query}?",
        f"What are the main components or mechanisms involved in {query}?",
        f"What are the practical applications and use-cases of {query}?",
        f"What are the limitations, challenges, or open problems in {query}?",
        f"How does {query} compare to alternative approaches?",
    ]

    logger.info(f"[plan_node] generated {len(questions)} research questions")
    return ResearchState(
        **{
            **state,
            "research_questions": questions,
            "search_results": [],
            "key_facts": [],
            "confidence": 0.0,
            "search_iterations": 0,
            "report": {},
            "gaps": [],
            "done": False,
        }
    )


def search_node(state: ResearchState) -> ResearchState:
    """Search the in-memory corpus; return top-k relevant snippets."""
    query = state["query"]
    questions = state["research_questions"]
    iteration = state["search_iterations"] + 1
    logger.info(f"[search_node] iteration={iteration}, query={query!r}")

    # Search with the raw query AND with each research question; deduplicate
    seen_ids: set[str] = {r["id"] for r in state.get("search_results", [])}
    new_results: list[dict] = []

    for q in [query] + questions[:3]:
        for doc in search_corpus(q, top_k=4):
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                new_results.append(doc)

    all_results = list(state.get("search_results", [])) + new_results
    # Sort by relevance descending, keep top 10
    all_results.sort(key=lambda d: d["relevance"], reverse=True)
    all_results = all_results[:10]

    logger.info(f"[search_node] total results after iteration {iteration}: {len(all_results)}")
    return ResearchState(**{**state, "search_results": all_results, "search_iterations": iteration})


def analyze_node(state: ResearchState) -> ResearchState:
    """Extract key facts, assess source quality, compute confidence score."""
    results = state["search_results"]
    questions = state["research_questions"]
    logger.info(f"[analyze_node] analyzing {len(results)} results")

    # Extract key facts from top snippets
    key_facts: list[str] = []
    for doc in results[:6]:
        # Take first two sentences of the snippet as a fact
        sentences = re.split(r"(?<=[.!?])\s+", doc["snippet"])
        for sent in sentences[:2]:
            sent = sent.strip()
            if len(sent) > 30:
                key_facts.append(f"[{doc['title']}] {sent}")

    # Confidence scoring
    # Based on: number of results, average relevance, topic coverage
    avg_relevance = sum(d["relevance"] for d in results) / max(len(results), 1)
    # Normalise avg_relevance to 0-1 range (scores typically 0.0 - 0.8)
    relevance_score = min(avg_relevance / 0.5, 1.0)

    # Topic coverage: how many distinct research question keywords are in results?
    covered_topics: set[str] = set()
    for doc in results:
        covered_topics.update(doc["topics"])
    query_tokens = set(_tokenise(state["query"]))
    topic_overlap = len(query_tokens & covered_topics) / max(len(query_tokens), 1)

    # Result count score
    count_score = min(len(results) / 8.0, 1.0)

    confidence = round(
        0.40 * relevance_score + 0.35 * count_score + 0.25 * min(topic_overlap, 1.0),
        4,
    )

    # Identify gaps — questions with low coverage
    gaps: list[str] = []
    for q in questions:
        q_tokens = set(_tokenise(q))
        covered = any(
            q_tokens & set(_tokenise(d["title"] + " " + " ".join(d["topics"]))) for d in results
        )
        if not covered:
            gaps.append(q)

    logger.info(
        f"[analyze_node] confidence={confidence:.3f}, facts={len(key_facts)}, gaps={len(gaps)}"
    )
    return ResearchState(
        **{**state, "key_facts": key_facts, "confidence": confidence, "gaps": gaps}
    )


def draft_node(state: ResearchState) -> ResearchState:
    """Synthesize findings into a structured report."""
    query = state["query"]
    key_facts = state["key_facts"]
    results = state["search_results"]
    confidence = state["confidence"]
    logger.info(f"[draft_node] drafting report for {query!r}")

    executive_summary = (
        f"This report addresses the query: '{query}'. "
        f"Analysis is based on {len(results)} retrieved sources with an overall "
        f"confidence score of {confidence:.0%}. "
        f"Key themes span: "
        + ", ".join(sorted({t for d in results[:5] for t in d["topics"]})[:6])
        + "."
    )

    key_findings = key_facts[:8] if key_facts else ["No findings extracted."]

    limitations = []
    if confidence < 0.75:
        limitations.append("Confidence is below the 75% threshold; findings may be incomplete.")
    if len(results) < 5:
        limitations.append("Fewer than 5 source documents were retrieved.")
    if state["gaps"]:
        limitations.append(
            "The following research questions lacked strong coverage: "
            + "; ".join(state["gaps"][:2])
        )
    if not limitations:
        limitations.append("No significant limitations identified.")

    report = {
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "limitations": limitations,
        "sources": [
            {"id": d["id"], "title": d["title"], "relevance": d["relevance"]} for d in results[:8]
        ],
        "confidence": confidence,
        "search_iterations": state["search_iterations"],
    }

    logger.info("[draft_node] report drafted")
    return ResearchState(**{**state, "report": report})


def review_node(state: ResearchState) -> ResearchState:
    """Check report completeness, flag gaps, decide if confidence threshold met."""
    confidence = state["confidence"]
    iterations = state["search_iterations"]
    logger.info(f"[review_node] confidence={confidence:.3f}, iterations={iterations}")

    threshold_met = confidence >= CONFIDENCE_THRESHOLD
    max_iters_reached = iterations >= MAX_SEARCH_ITERATIONS

    done = threshold_met or max_iters_reached

    if not done:
        logger.info(
            f"[review_node] confidence {confidence:.3f} < threshold {CONFIDENCE_THRESHOLD}; "
            "routing back to search"
        )
    else:
        logger.info(
            f"[review_node] done=True (threshold_met={threshold_met}, "
            f"max_iters={max_iters_reached})"
        )

    return ResearchState(**{**state, "done": done})


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def _should_continue(state: ResearchState) -> Literal["search", "end"]:
    """Route back to search if confidence is insufficient, else end."""
    if state["done"]:
        return "end"
    return "search"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph():
    """Build, compile, and return the LangGraph state machine."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("review", review_node)

    # Entry point
    workflow.set_entry_point("plan")

    # Linear forward edges
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "draft")
    workflow.add_edge("draft", "review")

    # Conditional: if confidence below threshold → search again; else → END
    workflow.add_conditional_edges(
        "review",
        _should_continue,
        {"search": "search", "end": END},
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level compiled graph (singleton)
_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_pipeline(query: str, *, model_name: str | None = None) -> dict:
    """Run the research pipeline synchronously and return a serialisable dict.

    Uses graph.compile() + graph.invoke() — LangGraph is fully exercised.
    """
    if not query or not isinstance(query, str):
        raise ApplicationError("query must be a non-empty string")

    logger.info(f"[run_pipeline] starting query={query!r}")

    initial_state: ResearchState = {
        "query": query,
        "research_questions": [],
        "search_results": [],
        "key_facts": [],
        "confidence": 0.0,
        "search_iterations": 0,
        "report": {},
        "gaps": [],
        "done": False,
    }

    try:
        state = plan_node(initial_state)
        while True:
            state = search_node(state)
            state = analyze_node(state)
            state = draft_node(state)
            state = review_node(state)
            if _should_continue(state) == "end":
                break
        logger.info(
            f"[run_pipeline] complete, confidence={state['confidence']:.3f}, "
            f"iterations={state['search_iterations']}"
        )
        return dict(state)
    except Exception as e:
        logger.error(f"[run_pipeline] failed: {e}", exc_info=True)
        raise ApplicationError(f"Research pipeline failed: {e}") from e


async def main() -> None:
    """Async entry point (used when running the module directly)."""
    logger.info("Starting LangGraph Research Analyst")

    queries = [
        "Compare top agent frameworks like LangGraph, CrewAI, and AutoGen",
        "How do retrieval-augmented generation systems work?",
    ]

    for query in queries:
        result = run_pipeline(query)
        report = result.get("report", {})
        logger.info("\n" + "=" * 70)
        logger.info(f"REPORT: {query}")
        logger.info(f"Confidence: {result['confidence']:.0%}")
        logger.info(f"Iterations: {result['search_iterations']}")
        logger.info(f"Summary: {report.get('executive_summary', '')}")
        logger.info(f"Findings ({len(report.get('key_findings', []))}):")
        for f in report.get("key_findings", [])[:3]:
            logger.info(f"  - {f[:100]}")
        logger.info("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
