# 01 - LangGraph Research Analyst

[![CI](https://github.com/milos-plavsic/langgraph-research-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/milos-plavsic/langgraph-research-analyst/actions/workflows/ci.yml)
[![Python3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

An advanced multi-agent research assistant that decomposes broad questions, gathers evidence, critiques weak claims, and produces cited reports with confidence scoring.

## System architecture

| Stage | Description |
|-------|-------------|
| Plan | Decompose the query into research questions |
| Corpus researcher | TF-IDF retrieval over the built-in technical corpus |
| Web researcher | Live Wikipedia summaries via agent-core |
| Merge | Deduplicate and rank combined evidence |
| Analyze | Score relevance and extract key facts |
| Critic | Validate coverage and adjust confidence |
| Draft | Structured report; set `LLM_API_KEY` to enable synthesis |
| Review | Confidence-gated loop with additional retrieval passes |

Libraries: [ml-core](https://github.com/milos-plavsic/ml-core), [agent-core](https://github.com/milos-plavsic/agent-core).

## Quickstart

```bash
make install
make run
make api          # http://127.0.0.1:8000/docs
make test
```

Docker API: `make docker-api` (Compose profile `api`).

## API

- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`
- End-to-end research run: `POST /v1/research` with JSON body `{"query":"..."}`

## Graph

`plan → corpus_researcher → web_researcher → merge_search → analyze → critic → draft → review`

The review node loops back to corpus retrieval when confidence stays below the configured threshold.
