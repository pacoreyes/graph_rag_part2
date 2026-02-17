# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from agent.state import State


def test_state_defaults():
    state = State()
    assert list(state.messages) == []
    assert state.retrieved_context == []
    assert state.strategy == ""
    assert state.query_embedding == []
    assert state.entities == []
    assert state.relationships == []
    assert state.community_reports == []
    assert state.chunk_evidence == []
    assert state.source_qids == []
    assert state.source_urls == {}


def test_state_with_context():
    state = State(retrieved_context=["doc1", "doc2"])
    assert state.retrieved_context == ["doc1", "doc2"]


def test_state_strategy():
    state = State(strategy="local")
    assert state.strategy == "local"


def test_state_query_embedding():
    state = State(query_embedding=[0.1, 0.2, 0.3])
    assert state.query_embedding == [0.1, 0.2, 0.3]


def test_state_entities():
    state = State(entities=[{"id": "e1", "name": "Einstein"}])
    assert len(state.entities) == 1
    assert state.entities[0]["name"] == "Einstein"


def test_state_source_urls():
    urls = {"Q1": {"name": "Alice", "wikipedia_url": "https://example.com"}}
    state = State(source_urls=urls)
    assert state.source_urls["Q1"]["name"] == "Alice"
