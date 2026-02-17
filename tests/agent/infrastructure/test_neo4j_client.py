# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for Neo4j async driver manager.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for Neo4j async driver manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.infrastructure.neo4j_client import Neo4jClient

pytestmark = pytest.mark.anyio


def test_neo4j_client_stores_config():
    """Verify constructor stores connection parameters."""
    client = Neo4jClient(
        uri="bolt://localhost:7687", username="neo4j", password="pw"
    )
    assert client._uri == "bolt://localhost:7687"
    assert client._username == "neo4j"
    assert client._password == "pw"
    assert client._drivers == {}
    assert client._loop_refs == {}


@patch("agent.infrastructure.neo4j_client.AsyncGraphDatabase")
async def test_get_driver_creates_driver(mock_adb):
    """Verify get_driver creates and caches a driver."""
    mock_driver = MagicMock()
    mock_adb.driver.return_value = mock_driver

    client = Neo4jClient(
        uri="bolt://localhost:7687", username="neo4j", password="pw"
    )
    driver = await client.get_driver()

    assert driver is mock_driver
    assert len(client._drivers) == 1
    assert len(client._loop_refs) == 1


@patch("agent.infrastructure.neo4j_client.AsyncGraphDatabase")
async def test_get_driver_returns_cached(mock_adb):
    """Verify repeated calls return the same cached driver."""
    mock_driver = MagicMock()
    mock_adb.driver.return_value = mock_driver

    client = Neo4jClient(
        uri="bolt://localhost:7687", username="neo4j", password="pw"
    )
    driver1 = await client.get_driver()
    driver2 = await client.get_driver()

    assert driver1 is driver2


@patch("agent.infrastructure.neo4j_client.AsyncGraphDatabase")
async def test_close_clears_all_drivers(mock_adb):
    """Verify close() clears drivers and loop refs."""
    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()
    mock_adb.driver.return_value = mock_driver

    client = Neo4jClient(
        uri="bolt://localhost:7687", username="neo4j", password="pw"
    )
    await client.get_driver()
    await client.close()

    assert client._drivers == {}
    assert client._loop_refs == {}
    mock_driver.close.assert_awaited_once()
