from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes.generation import (
    _check_faithfulness,
    _resolve_aku_legend,
    homogenize_context,
    planner,
    query_analyzer,
    synthesize_answer,
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


# --- query_analyzer ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_local(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "local", "target_entity_types": ["PERSON"]}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "local"
    assert result["target_entity_types"] == ["PERSON"]


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_global(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "global", "target_entity_types": []}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "global"
    assert result["target_entity_types"] == []


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_hybrid(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "hybrid", "target_entity_types": ["GENRE", "GROUP"]}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"
    assert result["target_entity_types"] == ["GENRE", "GROUP"]


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_invalid_falls_back_to_hybrid(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "unknown_strategy", "target_entity_types": []}'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_query_analyzer_strips_whitespace(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '  {"strategy": "local", "target_entity_types": []}  \n'

    result = await query_analyzer(_make_state(), _make_config())

    assert result["strategy"] == "local"


# --- planner ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_planner_local_fast_track(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "local", "is_fast_track": true, "plan": "Fast lookup.", "target_entity_types": ["GROUP"]}'

    result = await planner(_make_state(), _make_config())

    assert result["strategy"] == "local"
    assert result["is_fast_track"] is True
    assert result["plan"] == "Fast lookup."


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_planner_hybrid_normal(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"strategy": "hybrid", "is_fast_track": false, "plan": "Comprehensive search.", "target_entity_types": []}'

    result = await planner(_make_state(), _make_config())

    assert result["strategy"] == "hybrid"
    assert result["is_fast_track"] is False


# --- synthesize_answer ---


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_synthesize_answer_returns_ai_message(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"answer": "Einstein was a theoretical physicist [1].", "evidence": {"1": "Verbatim quote"}}'

    state = _make_state(
        entities=[{"name": "Einstein", "description": "Physicist", "pagerank": 1.0}],
    )
    result = await synthesize_answer(state, _make_config())

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "Einstein" in result["messages"][0].content
    assert "akus" in result
    assert len(result["akus"]) == 1
    assert result["akus"][0]["index"] == 1


@patch("agent.nodes.generation.gemini_client")
@patch("agent.nodes.generation.gemini_generate")
async def test_synthesize_answer_empty_context(mock_generate, mock_gc):
    mock_gc.get_client = AsyncMock(return_value=MagicMock())
    mock_generate.return_value = '{"answer": "I don\'t have enough context.", "evidence": {}}'

    state = _make_state()
    result = await synthesize_answer(state, _make_config())

    assert isinstance(result["messages"][0], AIMessage)
    assert result["akus"] == []


# --- Phase 3 Tests ---

def test_resolve_aku_legend_builds_correct_mapping():
    akus = [
        {"index": 4, "content": "Fact 1", "origin": "Graph DB", "method": "Search", "metadata": {"qid": "Q1"}},
        {"index": 6, "content": "Fact 2", "origin": "Vector DB", "method": "Summary", "metadata": {"article_id": "Q2", "type": "Text Chunk"}},
    ]
    source_urls = {
        "Q1": {"name": "Entity 1", "wikipedia_url": "http://link1"},
        "Q2": {"name": "Article 2", "wikipedia_url": "http://link2"},
    }
    answer = "Claim about fact 1 [4]. Another claim [6]."
    
    # We provide LLM evidence mapping
    evidence = {"4": "LLM Fact 1", "6": "LLM Quote 2"}
    
    updated_answer, legend = _resolve_aku_legend(answer, akus, source_urls, llm_evidence=evidence)
    
    # Check sequential re-indexing in answer
    assert "fact 1 [1]" in updated_answer
    assert "claim [2]" in updated_answer
    
    # Check legend
    assert "**Sources & Evidence Path:**" in legend
    assert "`[1]` LLM Fact 1 ([Entity 1](http://link1))" in legend
    assert '`[2]` "LLM Quote 2" ([Article 2](http://link2))' in legend


def test_resolve_aku_legend_skips_uncited():
    akus = [
        {"index": 1, "content": "Fact 1", "origin": "G", "method": "M"},
        {"index": 2, "content": "Fact 2", "origin": "G", "method": "M"},
    ]
    answer = "Only cited [1]"
    updated_answer, legend = _resolve_aku_legend(answer, akus, {})
    
    assert "[1]" in updated_answer
    assert "[1]" in legend
    assert "[2]" not in legend


def test_check_faithfulness_detects_hallucination():
    akus = [{"index": 1}]
    answer = "Hallucinated citation [2]"
    result = _check_faithfulness(answer, akus)
    assert result["is_faithful"] is False
    assert "Hallucinated indices" in result["issue"]


def test_check_faithfulness_detects_low_density():
    akus = [{"index": 1}]
    # Long answer with only one citation
    answer = "This is a very long answer that definitely needs more citations because it makes many claims but only has one at the very end even though it should have many more to be considered faithful to the context [1]. " * 5
    result = _check_faithfulness(answer, akus)
    assert result["is_faithful"] is False
    assert "Low citation density" in result["issue"]


def test_check_faithfulness_passes_valid():
    akus = [{"index": 1}, {"index": 2}]
    answer = "Claim [1]. Another [2]."
    result = _check_faithfulness(answer, akus)
    assert result["is_faithful"] is True


# --- homogenize_context tests ---


def test_homogenize_context_sorting_and_indexing():
    state = _make_state(
        entities=[
            {"name": "B", "description": "desc B", "pagerank": 0.5},
            {"name": "A", "description": "desc A", "pagerank": 1.0},
        ],
        relationships=[
            {"source_name": "S", "target_name": "T", "relationship": "R", "score": 0.9}
        ],
        chunk_evidence=[
            {"text": "chunk text", "score": 0.8}
        ],
        community_reports=[
            {"community_id": "c1", "title": "title", "summary": "summary", "score": 0.7}
        ],
        cypher_result=[{"count": 10}]
    )

    akus = homogenize_context(state)

    # Expected order: Entities (sorted), Relationships, Chunks, Communities, Cypher
    assert len(akus) == 6
    assert akus[0]["index"] == 1
    assert "Entity: A" in akus[0]["content"]  # Top PageRank
    assert akus[1]["index"] == 2
    assert "Entity: B" in akus[1]["content"]
    assert akus[2]["index"] == 3
    assert "Relationship: S --[R]--> T" in akus[2]["content"]
    assert akus[3]["index"] == 4
    assert "Text Evidence: chunk text" in akus[3]["content"]
    assert akus[4]["index"] == 5
    assert "Thematic Summary (Community c1)" in akus[4]["content"]
    assert akus[5]["index"] == 6
    assert "Database Fact: count: 10" in akus[5]["content"]


def test_homogenize_context_deduplication():
    # Identical content from different retrieval sources
    state = _make_state(
        entities=[
            {"name": "E", "description": "Desc", "pagerank": 1.0, "method": "M1"},
            {"name": "E", "description": "Desc", "pagerank": 0.9, "method": "M2"},
        ]
    )
    akus = homogenize_context(state)
    assert len(akus) == 1
    assert "M1 & M2" in akus[0]["method"]


def test_homogenize_context_empty():
    state = _make_state()
    akus = homogenize_context(state)
    assert akus == []


def test_homogenize_context_robustness_to_missing_keys():
    state = _make_state(
        entities=[{"name": "Einstein"}],  # Missing description, pagerank, origin
        relationships=[{"source_name": "S", "target_name": "T"}], # Missing relationship, score
        chunk_evidence=[{"text": "chunk"}], # Missing score
        community_reports=[{"summary": "sum"}], # Missing id, title, score
        cypher_result=[{"key": "val"}]
    )

    akus = homogenize_context(state)
    assert len(akus) == 5
    # Check if defaults are handled
    assert "Einstein" in akus[0]["content"]
    assert "No description" in akus[0]["content"]
    assert akus[0]["origin"] == "Graph DB" # Default in code
    assert akus[1]["content"] == "Relationship: S --[RELATED_TO]--> T"


def test_homogenize_context_sorting_with_none_values():
    state = _make_state(
        entities=[
            {"name": "A", "pagerank": None},
            {"name": "B", "pagerank": 0.5},
            {"name": "C", "pagerank": 1.0},
        ],
        chunk_evidence=[
            {"text": "T1", "score": 0.9},
            {"text": "T2", "score": None},
        ]
    )
    
    akus = homogenize_context(state)
    assert len(akus) == 5
    # Entities sorted: C (1.0), B (0.5), A (None/0)
    assert "Entity: C" in akus[0]["content"]
    assert "Entity: B" in akus[1]["content"]
    assert "Entity: A" in akus[2]["content"]


def test_homogenize_context_extreme_values():
    state = _make_state(
        entities=[
            {"name": "Inf", "pagerank": float("inf")},
            {"name": "Normal", "pagerank": 1.0},
        ]
    )
    akus = homogenize_context(state)
    assert "Entity: Inf" in akus[0]["content"]


def test_homogenize_context_large_dataset():
    state = _make_state(
        entities=[{"name": f"E{i}", "pagerank": i} for i in range(100)]
    )
    akus = homogenize_context(state)
    assert len(akus) == 100
    assert akus[0]["index"] == 1
    assert "Entity: E99" in akus[0]["content"]
    assert akus[99]["index"] == 100
    assert "Entity: E0" in akus[99]["content"]
