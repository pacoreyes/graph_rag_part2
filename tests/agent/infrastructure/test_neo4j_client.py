# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from unittest.mock import MagicMock, patch

from agent.infrastructure.neo4j_client import Neo4jClient


def test_neo4j_client_stores_config():
    client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="pw")
    assert client._uri == "bolt://localhost:7687"
    assert client._username == "neo4j"
    assert client._password == "pw"


@patch("agent.infrastructure.neo4j_client.AsyncGraphDatabase.driver")
def test_get_driver_creates_driver_once(mock_driver_factory):
    mock_driver = MagicMock()
    mock_driver_factory.return_value = mock_driver

    client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="pw")
    driver1 = client.get_driver()
    driver2 = client.get_driver()

    assert driver1 is driver2
    mock_driver_factory.assert_called_once_with(
        "bolt://localhost:7687", auth=("neo4j", "pw")
    )


@patch("agent.infrastructure.neo4j_client.AsyncGraphDatabase.driver")
def test_get_driver_returns_async_driver(mock_driver_factory):
    mock_driver = MagicMock()
    mock_driver_factory.return_value = mock_driver

    client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="pw")
    driver = client.get_driver()

    assert driver is mock_driver
