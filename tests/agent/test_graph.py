from langgraph.pregel import Pregel
from langgraph.graph import END

from agent.graph import (
    graph,
    route_by_strategy,
    route_from_join,
    route_after_synthesis,
)
from agent.state import State


def test_graph_is_compiled():
    assert isinstance(graph, Pregel)


def test_graph_has_expected_nodes():
    node_names = set(graph.nodes.keys())
    expected = {
        "planner",
        "embed_query",
        "entity_search",
        "neighborhood_expand",
        "chunk_search",
        "community_search",
        "community_members_search",
        "resolve_sources",
        "synthesize_answer",
        "answer_critic",
        "retrieval_evaluator",
        "nl_to_cypher",
        "join_retrieval",
    }
    assert expected.issubset(node_names)


def test_route_by_strategy_local():
    state = State(strategy="local")
    result = route_by_strategy(state)
    assert "entity_search" in result
    assert "nl_to_cypher" in result


def test_route_by_strategy_global():
    state = State(strategy="global")
    assert route_by_strategy(state) == ["community_search"]


def test_route_by_strategy_hybrid():
    state = State(strategy="hybrid")
    result = route_by_strategy(state)
    assert "entity_search" in result
    assert "community_search" in result
    assert "nl_to_cypher" in result


def test_route_from_join_fast_track():
    state = State(is_fast_track=True)
    assert route_from_join(state) == "resolve_sources"


def test_route_from_join_normal():
    state = State(is_fast_track=False)
    assert route_from_join(state) == "retrieval_evaluator"


def test_route_after_synthesis_fast_track():
    state = State(is_fast_track=True)
    assert route_after_synthesis(state) == END


def test_route_after_synthesis_normal():
    state = State(is_fast_track=False)
    assert route_after_synthesis(state) == "answer_critic"
