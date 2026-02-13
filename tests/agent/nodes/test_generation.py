from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes.generation import (
    _format_chunks,
    _format_community_reports,
    _format_entities,
    _format_relationships,
    _format_sources,
    query_analyzer,
    synthesize_answer,
)
from agent.state import State

pytestmark = pytest.mark.anyio


def _make_state(**overrides) -> State:
    defaults = {
        "messages": [HumanMessage(content="Who is Einstein?")],
    }
    defaults.update(overrides)
    return State(**defaults)


def _make_config(**configurable):
    return {"configurable": configurable} if configurable else {}


# --- query_analyzer ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_local(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = '{"strategy": "local", "target_entity_types": ["PERSON"]}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "local"
    assert result["target_entity_types"] == ["PERSON"]


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_global(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = '{"strategy": "global", "target_entity_types": []}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "global"
    assert result["target_entity_types"] == []


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_hybrid(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = '{"strategy": "hybrid", "target_entity_types": ["GENRE", "GROUP"]}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"
    assert result["target_entity_types"] == ["GENRE", "GROUP"]


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_invalid_falls_back_to_hybrid(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = '{"strategy": "unknown_strategy", "target_entity_types": []}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_strips_whitespace(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = '  {"strategy": "local", "target_entity_types": []}  \n'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "local"


# --- synthesize_answer ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_synthesize_answer_returns_ai_message(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = "Einstein was a theoretical physicist."

    state = _make_state(
        entities=[{"name": "Einstein", "description": "Physicist"}],
        source_urls={
            "Q937": {"name": "Einstein", "wikipedia_url": "https://example.com"}
        },
    )
    result = await synthesize_answer(state, _make_config())

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "Einstein" in result["messages"][0].content


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_synthesize_answer_empty_context(mock_generate, mock_gc):
    mock_gc.get_client.return_value = MagicMock()
    mock_generate.return_value = "I don't have enough context."

    state = _make_state()
    result = await synthesize_answer(state, _make_config())

    assert isinstance(result["messages"][0], AIMessage)


# --- Formatter tests ---


def test_format_entities_empty():
    assert _format_entities([]) == ""


def test_format_entities_with_data():
    result = _format_entities([{"name": "Einstein", "description": "Physicist"}])
    assert "## Entities" in result
    assert "**Einstein**" in result


def test_format_relationships_empty():
    assert _format_relationships([]) == ""


def test_format_relationships_with_data():
    result = _format_relationships(
        [
            {
                "source_name": "Einstein",
                "relationship": "BORN_IN",
                "target_name": "Ulm",
                "rel_description": "born in the city",
            }
        ]
    )
    assert "## Relationships" in result
    assert "BORN_IN" in result
    assert "born in the city" in result


def test_format_community_reports_empty():
    assert _format_community_reports([]) == ""


def test_format_community_reports_with_data():
    result = _format_community_reports(
        [{"community_id": "c1", "summary": "Physics community"}]
    )
    assert "## Community Reports" in result
    assert "Physics community" in result


def test_format_chunks_empty():
    assert _format_chunks([]) == ""


def test_format_chunks_with_data():
    result = _format_chunks([{"score": 0.85, "text": "Some text"}])
    assert "## Text Evidence" in result
    assert "0.850" in result


def test_format_sources_empty():
    assert _format_sources({}) == ""


def test_format_sources_with_url():
    result = _format_sources(
        {"Q1": {"name": "Alice", "wikipedia_url": "https://example.com"}}
    )
    assert "## Sources" in result
    assert "[Alice](https://example.com)" in result


def test_format_sources_without_url():
    result = _format_sources({"Q1": {"name": "Alice", "wikipedia_url": ""}})
    assert "- Alice" in result
