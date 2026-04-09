"""LangGraph agent graph for the Pickstape recommendation pipeline.

Fixed linear flow: START → router → recommendation → response → END
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes import create_nodes
from src.agent.state import AgentState

if TYPE_CHECKING:
    from src.recommender.engine import RecommendationEngine


def build_graph(
    engine: RecommendationEngine,
    *,
    router_llm_factory: Callable | None = None,
    response_llm_factory: Callable | None = None,
) -> CompiledStateGraph:
    """Build and compile the recommendation agent graph.

    Parameters
    ----------
    engine:
        A fully initialised RecommendationEngine.
    router_llm_factory:
        Optional override for the router LLM (for testing).
    response_llm_factory:
        Optional override for the response LLM (for testing).

    Returns
    -------
    A compiled LangGraph that accepts ``{"user_input": str}`` and
    produces ``{"response_text": str}`` (plus intermediate state).
    """
    router_node, recommendation_node, response_node = create_nodes(
        engine,
        router_llm_factory=router_llm_factory,
        response_llm_factory=response_llm_factory,
    )

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "recommendation")
    graph.add_edge("recommendation", "response")
    graph.add_edge("response", END)

    return graph.compile()
