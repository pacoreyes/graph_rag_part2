# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# LangGraph multi-strategy GraphRAG graph definition.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""LangGraph multi-strategy GraphRAG graph definition.

Routes queries through Local, Global, or Hybrid retrieval paths
based on LLM-driven query analysis, then synthesizes a cited answer.
"""

from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.nodes.generation import router, synthesize_answer
from agent.nodes.retrieval import (
    chunk_search,
    community_members_search,
    community_search,
    embed_query,
    entity_search,
    neighborhood_expand,
    resolve_sources,
)
from agent.nodes.structural import nl_to_cypher
from agent.state import State


def route_by_strategy(state: State) -> list[str]:
    """Route to retrieval nodes based on the classified strategy and fast track status.

    Args:
        state: Current graph state.

    Returns:
        list[str]: Node names to execute next.
    """
    if state.is_fast_track:
        return ["entity_search"]
        
    if state.strategy == "local":
        return ["entity_search", "nl_to_cypher"]
    if state.strategy == "drift":
        return ["entity_search"] # Drift uses semantic expansion, not Cypher
    if state.strategy == "global":
        return ["community_search"]
    if state.strategy == "structural":
        return ["nl_to_cypher"]
    return ["entity_search", "community_search", "nl_to_cypher"]


def route_after_retrieval(state: State) -> str:
    """Decide if we need deeper text search or can go to synthesis.

    Args:
        state: Current graph state.

    Returns:
        str: Next node to execute.
    """
    if state.is_fast_track:
        return "resolve_sources"
    return "chunk_search"


# Build the graph
builder = StateGraph(State, context_schema=Configuration)

# Nodes
builder.add_node("router", router)
builder.add_node("embed_query", embed_query)
builder.add_node("entity_search", entity_search)
builder.add_node("neighborhood_expand", neighborhood_expand)
builder.add_node("nl_to_cypher", nl_to_cypher)
builder.add_node("community_search", community_search)
builder.add_node("community_members_search", community_members_search)
builder.add_node("chunk_search", chunk_search)
builder.add_node("resolve_sources", resolve_sources)
builder.add_node("synthesize_answer", synthesize_answer)

# Flow: START -> router -> embed_query -> Parallel Retrieval
builder.add_edge(START, "router")
builder.add_edge("router", "embed_query")
builder.add_conditional_edges(
    "embed_query",
    route_by_strategy,
    {
        "entity_search": "entity_search",
        "community_search": "community_search",
        "nl_to_cypher": "nl_to_cypher",
    },
)

# --- Path A: Entity Discovery ---
builder.add_edge("entity_search", "neighborhood_expand")
builder.add_edge("neighborhood_expand", "chunk_search")

# --- Path B: Community Discovery ---
builder.add_edge("community_search", "community_members_search")
builder.add_edge("community_members_search", "chunk_search")

# --- Path C: Structural Discovery ---
builder.add_edge("nl_to_cypher", "chunk_search")

# --- Fast Track Override ---
# We use a custom join logic or simply route from the nodes
# For simplicity in this refactor, we allow chunk_search to be the convergence point
# but chunk_search itself will handle empty inputs gracefully.
# However, to be truly efficient on fast-track, we can add conditional edges after retrieval nodes.

builder.add_conditional_edges(
    "neighborhood_expand",
    route_after_retrieval,
    {"chunk_search": "chunk_search", "resolve_sources": "resolve_sources"}
)
builder.add_conditional_edges(
    "community_members_search",
    route_after_retrieval,
    {"chunk_search": "chunk_search", "resolve_sources": "resolve_sources"}
)
builder.add_conditional_edges(
    "nl_to_cypher",
    route_after_retrieval,
    {"chunk_search": "chunk_search", "resolve_sources": "resolve_sources"}
)

builder.add_edge("chunk_search", "resolve_sources")
builder.add_edge("resolve_sources", "synthesize_answer")
builder.add_edge("synthesize_answer", END)

graph = builder.compile()
