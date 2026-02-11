from langgraph.pregel import Pregel

from agent.graph import graph


def test_graph_is_compiled():
    assert isinstance(graph, Pregel)
