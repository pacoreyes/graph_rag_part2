# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for homogenize_context cross-node entity deduplication.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for homogenize_context cross-node entity deduplication."""

from langchain_core.messages import HumanMessage

from agent.nodes.generation import homogenize_context
from agent.state import State


def _make_state(**overrides) -> State:
    """Create a State with sensible defaults for testing."""
    defaults = {
        "messages": [HumanMessage(content="test query")],
        "query_embedding": [0.1] * 384,
        "strategy": "hybrid",
    }
    defaults.update(overrides)
    return State(**defaults)


def test_homogenize_deduplicates_entities_by_id():
    """Verify same entity from different retrieval paths is merged."""
    state = _make_state(
        entities=[
            {
                "id": "e1", "name": "David Bowie", "description": "British musician",
                "type": "PERSON", "qid": "Q5383", "score": 0.9,
                "pagerank": 0.5, "mention_count": 10,
                "origin": "Graph DB", "method": "Entity Search (Fulltext/Vector)",
            },
            {
                "id": "e1", "name": "David Bowie", "description": "British musician",
                "type": "PERSON", "qid": "Q5383", "score": 0.7,
                "pagerank": 0.5, "mention_count": 10,
                "origin": "Graph DB", "method": "Community Discovery (L2)",
            },
        ],
    )

    akus = homogenize_context(state)

    # Only one AKU for Bowie, with merged methods
    bowie_akus = [a for a in akus if "David Bowie" in a["content"]]
    assert len(bowie_akus) == 1
    assert "Entity Search" in bowie_akus[0]["method"]
    assert "Community Discovery" in bowie_akus[0]["method"]


def test_homogenize_keeps_best_score_on_merge():
    """Verify merged entity keeps the higher relevance score."""
    state = _make_state(
        entities=[
            {
                "id": "e1", "name": "Bowie", "description": "Musician",
                "type": "PERSON", "qid": "Q5383", "score": 0.6,
                "pagerank": 0.5, "mention_count": 10,
                "origin": "Graph DB", "method": "Entity Search (Fulltext/Vector)",
            },
            {
                "id": "e1", "name": "Bowie", "description": "Musician",
                "type": "PERSON", "qid": "Q5383", "score": 0.9,
                "pagerank": 0.5, "mention_count": 10,
                "origin": "Graph DB", "method": "Community Discovery (L2)",
            },
        ],
    )

    akus = homogenize_context(state)

    bowie_akus = [a for a in akus if "Bowie" in a["content"]]
    assert len(bowie_akus) == 1
    assert bowie_akus[0]["raw_relevance_score"] == 0.9


def test_homogenize_different_entities_not_merged():
    """Verify entities with different IDs remain separate."""
    state = _make_state(
        entities=[
            {
                "id": "e1", "name": "Bowie", "description": "Musician",
                "type": "PERSON", "qid": "Q5383", "score": 0.9,
                "pagerank": 0.5, "mention_count": 10,
                "origin": "Graph DB", "method": "Entity Search",
            },
            {
                "id": "e2", "name": "Eno", "description": "Producer",
                "type": "PERSON", "qid": "Q1234", "score": 0.8,
                "pagerank": 0.3, "mention_count": 5,
                "origin": "Graph DB", "method": "Entity Search",
            },
        ],
    )

    akus = homogenize_context(state)

    entity_akus = [a for a in akus if a["content"].startswith("Entity")]
    assert len(entity_akus) == 2
