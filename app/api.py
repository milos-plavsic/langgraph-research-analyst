from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.main import run_pipeline

app = FastAPI(title="LangGraph Research Analyst", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research question or topic")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/research")
def research(body: ResearchRequest) -> dict:
    state = run_pipeline(body.query)
    return asdict(state)
