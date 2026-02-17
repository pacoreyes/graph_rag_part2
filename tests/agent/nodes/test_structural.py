# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for the structural expert node.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for the structural expert node."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agent.nodes.structural import nl_to_cypher
from agent.state import State

pytestmark = pytest.mark.anyio


def _make_state(**overrides) -> State:
    """Create a State with sensible defaults for testing."""
    defaults = {
        "messages": [HumanMessage(content="Count all albums by The Beatles")],
        "strategy": "structural",
    }
    defaults.update(overrides)
    return State(**defaults)


def _make_config(**configurable):
    """Create a minimal LangGraph runtime config dict."""
    return {"configurable": configurable} if configurable else {}


@patch("agent.nodes.structural.gemini_client")
@patch("agent.nodes.structural.neo4j_client")
@patch("agent.nodes.structural.gemini_generate")
@patch("agent.nodes.structural.query_knowledge_graph", new_callable=AsyncMock)
async def test_nl_to_cypher_success(mock_kg, mock_generate, mock_neo4j, mock_gemini_client):
    """Test successful Cypher generation and execution."""
    # Setup Mocks
    # neo4j_client.get_driver() is async
    mock_neo4j.get_driver = AsyncMock(return_value=AsyncMock())
    
    # gemini_client.get_client() is async
    mock_gemini_client.get_client = AsyncMock(return_value=MagicMock())
    mock_gemini_client.get_schema_instruction.return_value = "Schema"
    
    # Mock Gemini response
    mock_generate.return_value = json.dumps({
        "cypher": "MATCH (a:Artist {name: 'The Beatles'})-[:RELEASED]->(al:Album) RETURN count(al) AS count"
    })
    
    # Mock KG responses (first for EXPLAIN, second for actual)
    mock_kg.side_effect = [
        [{"explain": "plan"}], # EXPLAIN
        [{"count": 13}] # Actual result
    ]

    state = _make_state()
    result = await nl_to_cypher(state, _make_config())

    assert result["cypher_error"] == ""
    assert result["generated_cypher"] == "MATCH (a:Artist {name: 'The Beatles'})-[:RELEASED]->(al:Album) RETURN count(al) AS count"
    assert result["cypher_result"] == [{"count": 13}]
    
    # Verify Gemini called with correct MIME type
    args, kwargs = mock_generate.call_args
    assert kwargs["response_mime_type"] == "application/json"


@patch("agent.nodes.structural.gemini_client")
@patch("agent.nodes.structural.neo4j_client")
@patch("agent.nodes.structural.gemini_generate")
@patch("agent.nodes.structural.query_knowledge_graph", new_callable=AsyncMock)
async def test_nl_to_cypher_retry_logic(mock_kg, mock_generate, mock_neo4j, mock_gemini_client):
    """Test that the node retries upon Cypher execution error."""
    # Setup Mocks
    mock_neo4j.get_driver = AsyncMock(return_value=AsyncMock())
    mock_gemini_client.get_client = AsyncMock(return_value=MagicMock())
    
    # Mock Gemini response (returns same query twice for simplicity)
    mock_generate.return_value = json.dumps({"cypher": "BAD QUERY"})
    
    # Mock KG responses: Fail first (EXPLAIN), Success second (EXPLAIN + Run)
    # The code runs EXPLAIN first. If that fails -> Exception -> Retry loop.
    # Second attempt: EXPLAIN passes -> Run passes.
    mock_kg.side_effect = [
        Exception("Syntax Error"), # 1st attempt EXPLAIN fails
        [{"explain": "plan"}],     # 2nd attempt EXPLAIN succeeds
        [{"count": 5}]             # 2nd attempt Run succeeds
    ]

    state = _make_state()
    result = await nl_to_cypher(state, _make_config())

    # Should succeed eventually
    assert result["cypher_error"] == ""
    assert result["cypher_result"] == [{"count": 5}]
    
    # Verify gemini_generate was called twice
    assert mock_generate.call_count == 2
