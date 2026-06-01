"""FastAPI application for the LangGraph Research Analyst.

Auth + rate limiting provided by ml-core:
  - APIKeyMiddleware  : rejects requests missing a valid X-API-Key header
    (no-op when API_KEY env var is unset, so local dev works without config)
  - RateLimiter       : token-bucket, 20 req/s burst per API-key
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from ml_core.exceptions import RateLimitExceeded
from ml_core.ratelimit import RateLimiter
from pydantic import BaseModel, Field

from app.main import run_pipeline
from finetune.extension import llm_training_guide

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="LangGraph Research Analyst", version="1.0.0")

# ---------------------------------------------------------------------------
# Rate limiter — 20 requests/second, burst of 30
# ---------------------------------------------------------------------------

_limiter = RateLimiter(rate=20.0, burst=30.0)


def _rate_limit(_key: str = "dev") -> None:
    """FastAPI dependency that enforces the per-key token-bucket rate limit."""
    client_key = _key or "anon"
    try:
        _limiter.acquire(client_key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Pydantic schema for the research request."""

    query: str = Field(..., min_length=1, description="Research question or topic")
    model_name: str | None = Field(default=None, description="Optional model override")


class ResearchResponse(BaseModel):
    """Pydantic schema for the research response."""

    query: str
    research_questions: list[str]
    key_facts: list[str]
    confidence: float
    search_iterations: int
    report: dict
    gaps: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health status.  No auth required."""
    return {"status": "ok"}


@app.post(
    "/v1/research",
    response_model=ResearchResponse,
)
async def research(body: ResearchRequest) -> dict:
    """Run the research pipeline and return a structured report.

    The pipeline runs a LangGraph state machine:
    plan → search → analyze → draft → review, with conditional
    re-search if confidence falls below threshold.

    Requires a valid ``X-API-Key`` header.  Rate-limited to 20 req/s
    per key (burst 30).
    """
    try:
        result = run_pipeline(body.query, model_name=body.model_name)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/finetune/playbook")
async def finetune_playbook() -> dict:
    """Return the LLM fine-tune playbook.  Requires auth."""
    return llm_training_guide()
