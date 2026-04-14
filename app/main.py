import os
from dataclasses import dataclass


@dataclass
class GraphState:
    query: str
    draft: str = ""
    confidence: float = 0.0


def planner_node(state: GraphState) -> GraphState:
    state.draft = f"Planned research steps for: {state.query}"
    return state


def critic_node(state: GraphState) -> GraphState:
    state.confidence = 0.82
    return state


def run_pipeline(query: str) -> GraphState:
    state = GraphState(query=query)
    state = planner_node(state)
    state = critic_node(state)
    return state


def main() -> None:
    query = os.getenv("DEMO_QUERY", "Compare top agent frameworks.")
    result = run_pipeline(query)
    print("LangGraph Research Analyst")
    print(f"Query: {result.query}")
    print(f"Draft: {result.draft}")
    print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    main()
