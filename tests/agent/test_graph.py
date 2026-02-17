# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from langgraph.pregel import Pregel
from langgraph.graph import END

from agent.graph import (
    graph,
    route_by_strategy,
    route_after_retrieval,
)
from agent.state import State


def test_graph_is_compiled():
    assert isinstance(graph, Pregel)


def test_graph_has_expected_nodes():
    node_names = set(graph.nodes.keys())
    expected = {
        "router",
        "embed_query",
        "entity_search",
        "neighborhood_expand",
        "chunk_search",
        "community_search",
        "community_members_search",
        "resolve_sources",
        "synthesize_answer",
        "nl_to_cypher",
    }
    assert expected.issubset(node_names)


def test_route_by_strategy_local():
    state = State(strategy="local", is_fast_track=False)
    result = route_by_strategy(state)
    assert "entity_search" in result
    assert "nl_to_cypher" in result


def test_route_by_strategy_fast_track():
    state = State(strategy="local", is_fast_track=True)
    assert route_by_strategy(state) == ["entity_search"]


def test_route_by_strategy_global():
    state = State(strategy="global", is_fast_track=False)
    assert route_by_strategy(state) == ["community_search"]


def test_route_by_strategy_hybrid():
    state = State(strategy="hybrid", is_fast_track=False)
    result = route_by_strategy(state)
    assert "entity_search" in result
    assert "community_search" in result
    assert "nl_to_cypher" not in result


def test_route_by_strategy_drift():
    state = State(strategy="drift", is_fast_track=False)
    result = route_by_strategy(state)
    assert result == ["entity_search"]
    assert "nl_to_cypher" not in result


def test_route_by_strategy_structural():
    state = State(strategy="structural", is_fast_track=False)
    result = route_by_strategy(state)
    assert result == ["nl_to_cypher"]


def test_route_after_retrieval_fast_track():
    state = State(is_fast_track=True)
    assert route_after_retrieval(state) == "resolve_sources"


def test_route_after_retrieval_normal():
    state = State(is_fast_track=False)
    assert route_after_retrieval(state) == "chunk_search"
