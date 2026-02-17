# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes.generation import (
    homogenize_context,
    router,
    synthesize_answer,
)
from agent.utils import (
    check_faithfulness,
    resolve_aku_legend,
)
from agent.state import State

pytestmark = pytest.mark.anyio


def _make_state(**overrides) -> State:
    defaults = {
        "messages": [HumanMessage(content="Who is Einstein?")],
        "source_urls": {}
    }
    defaults.update(overrides)
    return State(**defaults)


def _make_config(**configurable):
    return {"configurable": configurable} if configurable else {}


# --- router ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_router_local_fast_track(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "local", "is_fast_track": true, "plan": "Simple lookup.", "target_entity_types": ["PERSON"]}'

    result = await router(_make_state(), _make_config())

    assert result["strategy"] == "local"
    assert result["is_fast_track"] is True
    assert result["target_entity_types"] == ["PERSON"]


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_router_global(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "global", "is_fast_track": false, "plan": "Thematic.", "target_entity_types": []}'

    result = await router(_make_state(), _make_config())

    assert result["strategy"] == "global"
    assert result["is_fast_track"] is False


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_router_drift(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "drift", "is_fast_track": false, "plan": "Abstract impact.", "target_entity_types": []}'

    result = await router(_make_state(), _make_config())

    assert result["strategy"] == "drift"


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_router_hybrid(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "hybrid", "is_fast_track": false, "plan": "Complex query.", "target_entity_types": ["GENRE", "GROUP"]}'

    result = await router(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"
    assert result["target_entity_types"] == ["GENRE", "GROUP"]


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_router_invalid_falls_back_to_hybrid(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "unknown_strategy", "is_fast_track": false, "target_entity_types": []}'

    result = await router(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"


# --- homogenize_context ---


def test_homogenize_context_combines_sources():
    state = _make_state(
        entities=[
            {"name": "E1", "description": "Desc1", "score": 0.9, "qid": "Q1", "mention_count": 10},
            {"name": "E2", "description": "Desc2", "score": 0.8}
        ],
        relationships=[
            {"source_name": "E1", "target_name": "E2", "relationship": "R", "score": 0.7}
        ],
        chunk_evidence=[
            {"text": "Text1", "score": 0.6, "id": "C1"}
        ],
        community_reports=[
            {"title": "C1", "summary": "Sum1", "community_id": "1"}
        ]
    )

    akus = homogenize_context(state)

    # Check total count and presence of different types
    assert len(akus) >= 4
    content_strs = [aku["content"] for aku in akus]
    assert any("Entity (Entity): E1" in s for s in content_strs)
    assert any("Relationship: E1 --[R]--> E2" in s for s in content_strs)
    assert any("Text Evidence: Text1" in s for s in content_strs)
    assert any("Thematic Summary (Community 1): C1" in s for s in content_strs)

    # Check scoring/sorting (Summaries usually score high, followed by entities)
    assert akus[0]["importance"] > akus[-1]["importance"]


# --- synthesize_answer ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_synthesize_answer_returns_message(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"answer": "Einstein was a physicist [1].", "evidence": [{"index": 1, "content": "Physicist"}]}'

    state = _make_state()
    # Mock some context so we have something to cite
    state.entities = [{"name": "Einstein", "description": "Physicist", "qid": "Q1"}]
    
    result = await synthesize_answer(state, _make_config())

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert "Einstein was a physicist" in result["messages"][0].content


# --- helpers ---


def test_resolve_aku_legend_reindexes():
    answer = "Fact A [10]. Fact B [5]."
    akus = [
        {"index": 5, "content": "Info B", "origin": "Graph", "method": "Search", "metadata": {"name": "B"}},
        {"index": 10, "content": "Info A", "origin": "Vector", "method": "Search", "metadata": {"chunk_id": "A"}}
    ]
    urls = {}

    updated, legend = resolve_aku_legend(answer, akus, urls)

    # Order of appearance: [10] becomes [1], [5] becomes [2]
    assert "Fact A [1]." in updated
    assert "Fact B [2]." in updated
    assert "**Sources & Evidence Path:**" in legend
    # Check legend content
    assert "`[1]`" in legend and "Info A" in legend
    assert "`[2]`" in legend and "B" in legend


def test_check_faithfulness_detects_hallucination():
    answer = "Fact [99]."
    akus = [{"index": 1}]
    
    result = check_faithfulness(answer, akus)
    assert result["is_faithful"] is False
    assert "Hallucinated" in result["issue"]


def test_check_faithfulness_detects_low_density():
    # Long answer with no citations
    answer = "A" * 300
    akus = [{"index": 1}]
    
    result = check_faithfulness(answer, akus)
    assert result["is_faithful"] is False
    assert "Low citation density" in result["issue"]
