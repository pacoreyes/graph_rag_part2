"""LangGraph multi-strategy GraphRAG graph definition.

Routes queries through Local, Global, or Hybrid retrieval paths
based on LLM-driven query analysis, then synthesizes a cited answer.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.nodes.generation import planner, synthesize_answer
from agent.nodes.critic import answer_critic
from agent.nodes.evaluator import retrieval_evaluator
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
    """Route to retrieval nodes based on the classified strategy.

    Args:
        state: Current graph state with strategy field set.

    Returns:
        list[str]: Node names to execute next.
    """
    if state.strategy == "local" or state.strategy == "drift":
        return ["entity_search", "nl_to_cypher"]
    if state.strategy == "global":
        return ["community_search"]
    if state.strategy == "structural":
        return ["nl_to_cypher"]
    # hybrid: run all three paths for maximum coverage
    return ["entity_search", "community_search", "nl_to_cypher"]


def route_after_critic(state: State) -> str:
    """Decide whether to loop back for refinement or end.

    Args:
        state: Current graph state.

    Returns:
        str: Next node to execute.
    """
    if state.critique and state.iteration_count < 3:
        return "synthesize_answer"
    return END


def route_from_evaluator(state: State) -> str:
    """Decide if we need deeper text search or can go to synthesis.

    Args:
        state: Current graph state.

    Returns:
        str: Next node to execute.
    """
    if state.skip_deep_search:
        return "resolve_sources"
    return "chunk_search"


def route_from_join(state: State) -> str:
    """Decide if we skip evaluation for fast track.

    Args:
        state: Current graph state.

    Returns:
        str: Next node to execute.
    """
    if state.is_fast_track:
        return "resolve_sources"
    return "retrieval_evaluator"


def route_after_synthesis(state: State) -> str:
    """Decide if we skip critic for fast track.

    Args:
        state: Current graph state.

    Returns:
        str: Next node to execute.
    """
    if state.is_fast_track:
        return END
    return "answer_critic"


async def join_retrieval(state: State) -> dict[str, Any]:
    """No-op node to synchronize parallel retrieval paths.

    Args:
        state: Current graph state.

    Returns:
        dict[str, Any]: Empty update.
    """
    return {}


# Build the graph
builder = StateGraph(State, context_schema=Configuration)

# Generation nodes
builder.add_node("planner", planner)
builder.add_node("synthesize_answer", synthesize_answer)
builder.add_node("answer_critic", answer_critic)

# Retrieval nodes
builder.add_node("embed_query", embed_query)
builder.add_node("entity_search", entity_search)
builder.add_node("neighborhood_expand", neighborhood_expand)
builder.add_node("chunk_search", chunk_search)
builder.add_node("community_search", community_search)
builder.add_node("community_members_search", community_members_search)
builder.add_node("resolve_sources", resolve_sources)
builder.add_node("nl_to_cypher", nl_to_cypher)
builder.add_node("join_retrieval", join_retrieval)
builder.add_node("retrieval_evaluator", retrieval_evaluator)

# Flow: START -> planner -> embed_query -> conditional routing
builder.add_edge(START, "planner")
builder.add_edge("planner", "embed_query")
builder.add_conditional_edges(
    "embed_query",
    route_by_strategy,
    {
        "entity_search": "entity_search",
        "community_search": "community_search",
        "nl_to_cypher": "nl_to_cypher",
    },
)

# --- Path A: Entity Discovery (Local/Drift/Hybrid) ---
builder.add_edge("entity_search", "neighborhood_expand")
builder.add_edge("neighborhood_expand", "join_retrieval")

# --- Path B: Community Discovery (Global/Hybrid) ---
builder.add_edge("community_search", "community_members_search")
builder.add_edge("community_members_search", "join_retrieval")

# --- Path C: Structural Discovery (Structural/Hybrid) ---
builder.add_edge("nl_to_cypher", "join_retrieval")

# --- Converge & Surgical Retrieval ---
# join_retrieval acts as a synchronization barrier
builder.add_conditional_edges(
    "join_retrieval",
    route_from_join,
    {
        "retrieval_evaluator": "retrieval_evaluator",
        "resolve_sources": "resolve_sources"
    }
)

builder.add_conditional_edges(
    "retrieval_evaluator", 
    route_from_evaluator,
    {
        "chunk_search": "chunk_search",
        "resolve_sources": "resolve_sources"
    }
)

builder.add_edge("chunk_search", "resolve_sources")
builder.add_edge("resolve_sources", "synthesize_answer")

builder.add_conditional_edges(
    "synthesize_answer",
    route_after_synthesis,
    {
        "answer_critic": "answer_critic",
        END: END
    }
)

builder.add_conditional_edges("answer_critic", route_after_critic)

graph = builder.compile()
