"""LangGraph multi-strategy GraphRAG graph definition.

Routes queries through Local, Global, or Hybrid retrieval paths
based on LLM-driven query analysis, then synthesizes a cited answer.
"""

from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.nodes.generation import query_analyzer, synthesize_answer
from agent.nodes.retrieval import (
    chunk_search,
    community_members_search,
    community_search,
    embed_query,
    entity_search,
    neighborhood_expand,
    resolve_sources,
)
from agent.state import State


def route_by_strategy(state: State) -> list[str]:
    """Route to retrieval nodes based on the classified strategy.

    Args:
        state: Current graph state with strategy field set.

    Returns:
        list[str]: Node names to execute next.
    """
    if state.strategy == "local":
        return ["entity_search"]
    if state.strategy == "global":
        return ["community_search"]
    # hybrid: run both paths
    return ["entity_search", "community_search"]


# Build the graph
builder = StateGraph(State, context_schema=Configuration)

# Generation nodes
builder.add_node("query_analyzer", query_analyzer)
builder.add_node("synthesize_answer", synthesize_answer)

# Retrieval nodes
builder.add_node("embed_query", embed_query)
builder.add_node("entity_search", entity_search)
builder.add_node("neighborhood_expand", neighborhood_expand)
builder.add_node("chunk_search", chunk_search)
builder.add_node("community_search", community_search)
builder.add_node("community_members_search", community_members_search)
builder.add_node("resolve_sources", resolve_sources)

# Flow: START -> query_analyzer -> embed_query -> conditional routing
builder.add_edge(START, "query_analyzer")
builder.add_edge("query_analyzer", "embed_query")
builder.add_conditional_edges("embed_query", route_by_strategy)

# Local path: entity_search -> neighborhood_expand -> chunk_search -> resolve
builder.add_edge("entity_search", "neighborhood_expand")
builder.add_edge("neighborhood_expand", "chunk_search")
builder.add_edge("chunk_search", "resolve_sources")

# Global path: community_search -> community_members_search -> resolve
builder.add_edge("community_search", "community_members_search")
builder.add_edge("community_members_search", "resolve_sources")

# Converge: resolve_sources -> synthesize_answer -> END
builder.add_edge("resolve_sources", "synthesize_answer")
builder.add_edge("synthesize_answer", END)

graph = builder.compile()
