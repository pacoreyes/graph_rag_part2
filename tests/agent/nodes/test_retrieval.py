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


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_entity_search_prunes_low_score_before_dedup(mock_kg, mock_neo4j):
    """Verify score pruning uses global top score, not post-dedup top score."""
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())

    # First call (fulltext): high-score entity
    # Second call (vector): low-score entity that should be pruned
    mock_kg.side_effect = [
        [
            {
                "id": "e1",
                "name": "High Score",
                "description": "Top result",
                "qid": "Q1",
                "score": 1.0,
                "pagerank": 0.01,
            },
        ],
        [
            {
                "id": "e2",
                "name": "Low Score",
                "description": "Should be pruned",
                "qid": "Q2",
                "score": 0.5,
                "pagerank": 100.0,
            },
        ],
    ]

    state = _make_state()
    result = await entity_search(state, _make_config())

    # e2 has score 0.5 which is < 1.0 * 0.8 = 0.8, so it should be pruned
    # even though it has very high pagerank
    entity_names = [e["name"] for e in result["entities"]]
    assert "High Score" in entity_names
    assert "Low Score" not in entity_names


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


# --- entity_search: target_entity_types soft filter ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_entity_search_filters_by_target_types_when_enough(mock_kg, mock_neo4j):
    """Verify entity_search keeps only matching types when enough results."""
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())
    mock_kg.return_value = [
        {"id": "e1", "name": "Bowie", "description": "Singer", "qid": "Q5383",
         "score": 0.9, "pagerank": 0.5, "type": "PERSON"},
        {"id": "e2", "name": "Berlin", "description": "City", "qid": "Q64",
         "score": 0.85, "pagerank": 0.4, "type": "CITY"},
        {"id": "e3", "name": "Eno", "description": "Producer", "qid": "Q1234",
         "score": 0.8, "pagerank": 0.3, "type": "PERSON"},
        {"id": "e4", "name": "Kraftwerk", "description": "Group", "qid": "Q5678",
         "score": 0.75, "pagerank": 0.2, "type": "GROUP"},
        {"id": "e5", "name": "Iggy", "description": "Musician", "qid": "Q9012",
         "score": 0.75, "pagerank": 0.2, "type": "PERSON"},
    ]

    state = _make_state(target_entity_types=["PERSON"])
    result = await entity_search(state, _make_config(retrieval_k=3))

    # 3 PERSON entities found >= retrieval_k=3, so CITY and GROUP filtered out
    types = [e["type"] for e in result["entities"]]
    assert "CITY" not in types
    assert "GROUP" not in types
    assert all(t == "PERSON" for t in types)


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_entity_search_keeps_all_when_filtered_too_few(mock_kg, mock_neo4j):
    """Verify entity_search keeps all types when filtered set is too small."""
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())
    mock_kg.return_value = [
        {"id": "e1", "name": "Bowie", "description": "Singer", "qid": "Q5383",
         "score": 0.9, "pagerank": 0.5, "type": "PERSON"},
        {"id": "e2", "name": "Berlin", "description": "City", "qid": "Q64",
         "score": 0.85, "pagerank": 0.4, "type": "CITY"},
    ]

    state = _make_state(target_entity_types=["PERSON"])
    result = await entity_search(state, _make_config(retrieval_k=5))

    # Only 1 PERSON < retrieval_k=5, so all entities kept
    assert len(result["entities"]) == 2


# --- community_members_search: target_entity_types soft filter ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_community_members_filters_by_target_types(mock_kg, mock_neo4j):
    """Verify community_members_search applies soft type filter."""
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())
    mock_kg.return_value = [
        {"id": "e1", "name": "Bowie", "type": "PERSON", "qid": "Q5383"},
        {"id": "e2", "name": "Berlin", "type": "CITY", "qid": "Q64"},
        {"id": "e3", "name": "Eno", "type": "PERSON", "qid": "Q1234"},
    ]

    state = _make_state(
        community_reports=[{"community_id": "c1", "level": 2}],
        target_entity_types=["PERSON"],
    )
    result = await community_members_search(state, _make_config())

    assert len(result["entities"]) == 2
    assert all(e["type"] == "PERSON" for e in result["entities"])
    assert "Q64" not in result["source_qids"]


# --- chunk_search: graduated fallback ---


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_chunk_search_graduated_fallback(mock_embed, mock_vs, mock_pc):
    """Verify chunk_search tries progressively looser filters."""
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024

    # First two calls return empty (strict filters fail), third succeeds
    mock_vs.side_effect = [
        {"matches": []},  # Level 0: entity_type AND article_id — fails
        {"matches": []},  # Level 1: article_id only — fails
        {"matches": [     # Level 2: entity_type only — succeeds
            {
                "id": "c1", "score": 0.8,
                "metadata": {"text": "Found it", "article_id": "Q1",
                              "pagerank": 1.0, "entity_type": "PERSON"},
            }
        ]},
    ]

    state = _make_state(
        entities=[{"qid": "Q1"}],
        target_entity_types=["PERSON"],
    )
    result = await chunk_search(state, _make_config(retrieval_k=5))

    assert len(result["chunk_evidence"]) == 1
    assert mock_vs.call_count == 3


# --- neighborhood_expand: parallel execution ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_neighborhood_expand_parallel_multiple_entities(mock_kg, mock_neo4j):
    """Verify multiple entities at same depth are queried concurrently."""
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())

    # Each entity gets different neighbors
    mock_kg.side_effect = [
        [{"id": "n1", "name": "Neighbor1", "qid": "Q10", "score": 0.9,
          "source_name": "A", "relationship": "REL", "type": "REL"}],
        [{"id": "n2", "name": "Neighbor2", "qid": "Q20", "score": 0.85,
          "source_name": "B", "relationship": "REL", "type": "REL"}],
    ]

    state = _make_state(entities=[
        {"id": "e1", "name": "A"},
        {"id": "e2", "name": "B"},
    ])
    result = await neighborhood_expand(state, _make_config())

    assert len(result["relationships"]) == 2
    assert mock_kg.call_count == 2
    assert set(result["source_qids"]) == {"Q10", "Q20"}


# --- community_members_search: parallel execution ---


@patch("agent.nodes.retrieval.neo4j_client")
@patch("agent.nodes.retrieval.query_knowledge_graph", new_callable=AsyncMock)
async def test_community_members_parallel_multiple_communities(mock_kg, mock_neo4j):
    """Verify multiple communities are queried concurrently."""
    mock_neo4j.get_driver = AsyncMock(return_value=MagicMock())

    mock_kg.side_effect = [
        [{"id": "e1", "name": "Member1", "type": "PERSON", "qid": "Q1"}],
        [{"id": "e2", "name": "Member2", "type": "GROUP", "qid": "Q2"}],
    ]

    state = _make_state(
        community_reports=[
            {"community_id": "c1", "level": 2},
            {"community_id": "c2", "level": 1},
        ],
    )
    result = await community_members_search(state, _make_config())

    assert len(result["entities"]) == 2
    assert mock_kg.call_count == 2


# --- community_search: batched multi-level ---


@patch("agent.nodes.retrieval.pinecone_client")
@patch("agent.nodes.retrieval.vector_search")
@patch("agent.nodes.retrieval.pinecone_embed")
async def test_community_search_batches_multi_level(mock_embed, mock_vs, mock_pc):
    """Verify multi-level search uses a single $or call instead of two."""
    mock_pc.get_client = AsyncMock(return_value=MagicMock())
    mock_embed.return_value = [0.1] * 1024
    mock_vs.return_value = {"matches": []}

    state = _make_state(strategy="global")
    await community_search(state, _make_config(community_level=2))

    # Should be a single call with $or filter, not two separate calls
    assert mock_vs.call_count == 1
    call_kwargs = mock_vs.call_args[1]
    assert "$or" in call_kwargs["filter_dict"]
