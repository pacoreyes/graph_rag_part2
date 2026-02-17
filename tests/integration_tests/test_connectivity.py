# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Live connectivity tests for external services.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Live connectivity tests for external services.

These tests hit real APIs using credentials from .env.
Run explicitly: python -m pytest tests/integration_tests/test_connectivity.py -v -m integration
"""

import pytest

from agent.infrastructure.clients import gemini_client, neo4j_client, pinecone_client
from agent.settings import settings

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_neo4j_connectivity():
    """Verify Neo4j Aura connection with a simple ping query."""
    driver = neo4j_client.get_driver()
    async with driver.session() as session:
        result = await session.run("RETURN 1 AS ping")
        record = await result.single()
        assert record["ping"] == 1
    await neo4j_client.close()


def test_pinecone_connectivity():
    """Verify Pinecone connection by listing indexes."""
    pc = pinecone_client.get_client()
    indexes = pc.list_indexes()
    assert indexes is not None
    index_names = [idx.name for idx in indexes]
    assert len(index_names) >= 0  # just confirm the call succeeds


def test_gemini_connectivity():
    """Verify Gemini API connection with a simple generate call."""
    client = gemini_client.get_client()
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-lite", contents="Reply with exactly: pong"
    )
    assert response.text is not None
    assert len(response.text) > 0
