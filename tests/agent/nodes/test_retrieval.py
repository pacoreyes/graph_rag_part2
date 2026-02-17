# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Create a State with sensible defaults for testing.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agent.nodes.retrieval import (
    chunk_search,
    community_members_search,
    community_search,
    embed_query,
    entity_search,
    neighborhood_expand,
    resolve_sources,
)
from agent.state import State

pytestmark = pytest.mark.anyio


def _make_state(**overrides) -> State:
    """Create a State with sensible defaults for testing."""
    defaults = {
        "messages": [HumanMessage(content="Who is Albert Einstein?")],
        "query_embedding": [0.1] * 768,
        "strategy": "local",
    }
    defaults.update(overrides)
    return State(**defaults)


def _make_config(**configurable):
    """Create a minimal LangGraph runtime config dict."""
    return {"configurable": configurable} if configurable else {}


# --- embed_query ---


@patch("agent.nodes.retrieval.hf_embedding_client")
async def test_embed_query_returns_embedding(mock_hf_client):
    mock_hf_client.embed = AsyncMock(return_value=[0.1] * 384)

    with patch("agent.nodes.retrieval.process_embedding", return_value=[0.5] * 384):
        state = _make_state()
        result = await embed_query(state, _make_config())

    assert "query_embedding" in result
    assert len(result["query_embedding"]) == 384


# --- entity_search ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_entity_search_returns_entities(mock_kg, mock_neo4j):
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())
    mock_kg.return_value = [
        {
            "id": "e1",
            "name": "Einstein",
            "description": "Physicist",
            "qid": "Q937",
            "score": 0.9,
        },
        {
            "id": "e1",
            "name": "Einstein",
            "description": "Physicist",
            "qid": "Q937",
            "score": 0.8,
        },
        {
            "id": "e2",
            "name": "Physics",
            "description": "Science",
            "qid": None,
            "score": 0.85,
        },
    ]

    state = _make_state()
    result = await entity_search(state, _make_config())

    assert len(result["entities"]) == 2  # deduplicated
    assert result["source_qids"] == ["Q937"]  # only non-None qids
    assert result["entities"][0]["origin"] == "Graph DB"
    assert "method" in result["entities"][0]


# --- neighborhood_expand ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_neighborhood_expand_traverses_entities(mock_kg, mock_neo4j):
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())
    mock_kg.return_value = [
        {
            "id": "e3",
            "name": "Ulm",
            "source_id": "e1",
            "source_name": "Einstein",
            "relationship": "ORIGINATES_FROM",
            "rel_description": "born in",
            "qid": "Q3012",
            "score": 0.9,
            "type": "ORIGINATES_FROM"
        },
    ]

    state = _make_state(entities=[{"id": "e1", "name": "Einstein"}])
    result = await neighborhood_expand(state, _make_config())

    assert len(result["relationships"]) == 1
    assert result["source_qids"] == ["Q3012"]
    assert result["relationships"][0]["origin"] == "Graph DB"
    assert result["relationships"][0]["type"] == "ORIGINATES_FROM"


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_neighborhood_expand_skips_entities_without_id(mock_kg, mock_neo4j):
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())

    state = _make_state(entities=[{"name": "NoId"}])
    result = await neighborhood_expand(state, _make_config())

    assert result["relationships"] == []
    mock_kg.assert_not_called()


# --- chunk_search ---


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_chunk_search_returns_chunks(mock_embed, mock_vs, mock_pc):
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024
    mock_vs.return_value = {
        "matches": [
            {
                "id": "c1",
                "score": 0.5,
                "metadata": {
                    "text": "High authority",
                    "article_id": "Q1",
                    "pagerank": 10.0,
                    "entity_type": "PERSON"
                },
            },
            {
                "id": "c2",
                "score": 0.8,
                "metadata": {
                    "text": "Low authority",
                    "article_id": "Q2",
                    "pagerank": 0.1,
                    "entity_type": "PERSON"
                },
            },
        ]
    }

    state = _make_state(entities=[{"qid": "Q1"}, {"qid": "Q2"}])
    result = await chunk_search(state, _make_config(retrieval_k=2))

    assert len(result["chunk_evidence"]) == 2
    assert result["chunk_evidence"][0]["id"] == "c1"
    assert result["chunk_evidence"][0]["origin"] == "Vector DB"
    assert result["chunk_evidence"][0]["type"] == "Text Chunk"


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_chunk_search_no_filter_when_no_qids(mock_embed, mock_vs, mock_pc):
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024
    mock_vs.return_value = {"matches": []}

    state = _make_state(entities=[])
    await chunk_search(state, _make_config())

    # The fallback call (second call) has filter_dict=None
    call_kwargs = mock_vs.call_args[1]
    assert call_kwargs["filter_dict"] is None


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_chunk_search_applies_surgical_filters(mock_embed, mock_vs, mock_pc):
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024
    mock_vs.return_value = {"matches": []}

    state = _make_state(
        entities=[{"qid": "Q1"}],
        community_reports=[{"community_id": 42}],
        target_entity_types=["PERSON"],
        strategy="hybrid"
    )
    await chunk_search(state, _make_config())

    first_call_kwargs = mock_vs.call_args_list[0][1]
    filter_dict = first_call_kwargs["filter_dict"]
    
    assert "$and" in filter_dict


# --- community_search ---


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_community_search_returns_reports(mock_embed, mock_vs, mock_pc):
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024
    mock_vs.return_value = {
        "matches": [
            {
                "id": "r1",
                "score": 0.9,
                "metadata": {
                    "text": "Physics community",
                    "community_id": "comm-1",
                    "level": 2,
                },
            },
        ]
    }

    state = _make_state()
    result = await community_search(state, _make_config())

    assert len(result["community_reports"]) == 1
    assert result["community_reports"][0]["community_id"] == "comm-1"
    assert result["community_reports"][0]["origin"] == "Vector DB"


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_community_search_applies_level_filter(mock_embed, mock_vs, mock_pc):
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024
    mock_vs.return_value = {"matches": []}

    state = _make_state()
    await community_search(state, _make_config(community_level=3))

    call_kwargs = mock_vs.call_args[1]
    assert call_kwargs["filter_dict"] == {"level": {"$eq": 3}}


# --- community_members_search ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_community_members_returns_entities(mock_kg, mock_neo4j):
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())
    mock_kg.return_value = [
        {"id": "e5", "name": "Bohr", "description": "Physicist", "qid": "Q4985"},
    ]

    state = _make_state(
        community_reports=[{"community_id": "comm-1", "summary": "test", "level": 2}]
    )
    result = await community_members_search(state, _make_config())

    assert len(result["entities"]) == 1
    assert result["source_qids"] == ["Q4985"]
    assert result["entities"][0]["origin"] == "Graph DB"


# --- resolve_sources ---


@patch("agent.nodes.retrieval.resolve_source_urls")
async def test_resolve_sources_calls_resolver(mock_resolver):
    mock_resolver.return_value = {
        "Q937": {
            "name": "Einstein",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Einstein",
        }
    }

    state = _make_state(source_qids=["Q937"])
    result = await resolve_sources(state, _make_config())

    assert "Q937" in result["source_urls"]
