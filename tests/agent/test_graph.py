from langgraph.pregel import Pregel

from agent.graph import graph, route_by_strategy
from agent.state import State


def test_graph_is_compiled():
    assert isinstance(graph, Pregel)


def test_graph_has_expected_nodes():
    node_names = set(graph.nodes.keys())
    expected = {
        "query_analyzer",
        "embed_query",
        "entity_search",
        "neighborhood_expand",
        "chunk_search",
        "community_search",
        "community_members_search",
        "resolve_sources",
        "synthesize_answer",
    }
    assert expected.issubset(node_names)


def test_route_by_strategy_local():
    state = State(strategy="local")
    assert route_by_strategy(state) == ["entity_search"]


def test_route_by_strategy_global():
    state = State(strategy="global")
    assert route_by_strategy(state) == ["community_search"]


def test_route_by_strategy_hybrid():
    state = State(strategy="hybrid")
    result = route_by_strategy(state)
    assert "entity_search" in result
    assert "community_search" in result


def test_route_by_strategy_unknown_defaults_to_hybrid():
    state = State(strategy="something_else")
    result = route_by_strategy(state)
    assert "entity_search" in result
    assert "community_search" in result
