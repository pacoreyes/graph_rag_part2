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


@patch("agent.nodes.retrieval.nomic_client")
async def test_embed_query_returns_embedding(mock_nomic_client):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_nomic_client.get_model.return_value = mock_model
    mock_nomic_client.get_tokenizer.return_value = mock_tokenizer

    with patch("agent.nodes.retrieval.nomic_embed", return_value=[0.5] * 384):
        state = _make_state()
        result = await embed_query(state, _make_config())

    assert "query_embedding" in result
    assert len(result["query_embedding"]) == 384


# --- entity_search ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_entity_search_returns_entities(mock_kg, mock_neo4j):
    mock_neo4j.get_driver.return_value = MagicMock()
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


# --- neighborhood_expand ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_neighborhood_expand_traverses_entities(mock_kg, mock_neo4j):
    mock_neo4j.get_driver.return_value = MagicMock()
    mock_kg.return_value = [
        {
            "source_id": "e1",
            "source_name": "Einstein",
            "relationship": "BORN_IN",
            "rel_description": "born in",
            "target_id": "e3",
            "target_name": "Ulm",
            "qid": "Q3012",
        },
    ]

    state = _make_state(entities=[{"id": "e1", "name": "Einstein"}])
    result = await neighborhood_expand(state, _make_config())

    assert len(result["relationships"]) == 1
    assert result["source_qids"] == ["Q3012"]


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_neighborhood_expand_skips_entities_without_id(mock_kg, mock_neo4j):
    mock_neo4j.get_driver.return_value = MagicMock()

    state = _make_state(entities=[{"name": "NoId"}])
    result = await neighborhood_expand(state, _make_config())

    assert result["relationships"] == []
    mock_kg.assert_not_called()


# --- chunk_search ---


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
async def test_chunk_search_returns_chunks(mock_vs, mock_pc):
    mock_pc.get_client.return_value = MagicMock()
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

    # Re-ranking:
    # c1: 0.5 * log1p(10) = 0.5 * 2.39 = 1.195
    # c2: 0.8 * log1p(0.1) = 0.8 * 0.095 = 0.076
    # So c1 should be first even with lower initial score
    assert len(result["chunk_evidence"]) == 2
    assert result["chunk_evidence"][0]["id"] == "c1"
    assert result["chunk_evidence"][0]["score"] > 1.0
    assert result["chunk_evidence"][1]["id"] == "c2"
    assert result["chunk_evidence"][0]["entity_type"] == "PERSON"


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
async def test_chunk_search_no_filter_when_no_qids(mock_vs, mock_pc):
    mock_pc.get_client.return_value = MagicMock()
    mock_vs.return_value = {"matches": []}

    state = _make_state(entities=[])
    await chunk_search(state, _make_config())

    call_kwargs = mock_vs.call_args[1]
    assert call_kwargs["filter_dict"] is None


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
async def test_chunk_search_applies_surgical_filters(mock_vs, mock_pc):
    mock_pc.get_client.return_value = MagicMock()
    mock_vs.return_value = {"matches": []}

    state = _make_state(
        entities=[{"qid": "Q1"}],
        community_reports=[{"community_id": 42}],
        target_entity_types=["PERSON"],
        strategy="hybrid"
    )
    await chunk_search(state, _make_config())

    # Inspect the first call (filtered search)
    first_call_kwargs = mock_vs.call_args_list[0][1]
    filter_dict = first_call_kwargs["filter_dict"]
    
    # Check structure: $and with entity_type and $or (article_id, community_id)
    assert "$and" in filter_dict
    assert {"entity_type": {"$in": ["PERSON"]}} in filter_dict["$and"]
    or_filter = next(f for f in filter_dict["$and"] if "$or" in f)["$or"]
    assert {"article_id": {"$in": ["Q1"]}} in or_filter
    assert {"community_id": {"$in": [42]}} in or_filter


# --- community_search ---


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
async def test_community_search_returns_reports(mock_vs, mock_pc):
    mock_pc.get_client.return_value = MagicMock()
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


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
async def test_community_search_applies_level_filter(mock_vs, mock_pc):
    mock_pc.get_client.return_value = MagicMock()
    mock_vs.return_value = {"matches": []}

    state = _make_state()
    await community_search(state, _make_config(community_level=3))

    call_kwargs = mock_vs.call_args[1]
    assert call_kwargs["filter_dict"] == {"level": {"$eq": 3}}


# --- community_members_search ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_community_members_returns_entities(mock_kg, mock_neo4j):
    mock_neo4j.get_driver.return_value = MagicMock()
    mock_kg.return_value = [
        {"id": "e5", "name": "Bohr", "description": "Physicist", "qid": "Q4985"},
    ]

    state = _make_state(
        community_reports=[{"community_id": "comm-1", "summary": "test", "level": 2}]
    )
    result = await community_members_search(state, _make_config())

    assert len(result["entities"]) == 1
    assert result["source_qids"] == ["Q4985"]


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
    mock_resolver.assert_called_once()
