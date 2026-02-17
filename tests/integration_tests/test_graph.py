# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Test graph can be invoked with a proper State-compatible input.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import pytest
from langchain_core.messages import HumanMessage
from langgraph.pregel import Pregel

from agent.graph import graph

pytestmark = pytest.mark.anyio


async def test_graph_invocation_with_state():
    """Test graph can be invoked with a proper State-compatible input."""
    inputs = {"messages": [HumanMessage(content="hello")]}
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert "messages" in res


def test_graph_is_pregel_instance():
    assert isinstance(graph, Pregel)
