from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.knowledge_graph import (
    query_knowledge_graph,
    sanitize_lucene_query,
)

pytestmark = pytest.mark.anyio


def test_sanitize_lucene_query_escapes_special_chars():
    # Test common special characters
    raw = 'where is Kraftwerk from?"'
    sanitized = sanitize_lucene_query(raw)
    assert 'Kraftwerk from\\?\\"' in sanitized

    # Test Lucene operators
    raw = 'band (The Cure) + movement'
    sanitized = sanitize_lucene_query(raw)
    assert '\\(The Cure\\) \\+ movement' in sanitized

    # Test logical operators
    raw = 'techno AND germany NOT UK'
    sanitized = sanitize_lucene_query(raw)
    assert 'techno \\AND germany \\NOT UK' in sanitized


def _make_mock_driver(return_data):
    """Create a mock Neo4j driver with given return data."""
    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=return_data)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_ctx

    return mock_driver, mock_session


async def test_query_knowledge_graph_returns_data():
    mock_driver, mock_session = _make_mock_driver([{"n": "node1"}, {"n": "node2"}])

    result = await query_knowledge_graph("MATCH (n) RETURN n", driver=mock_driver)

    assert result == [{"n": "node1"}, {"n": "node2"}]
    mock_session.run.assert_called_once_with("MATCH (n) RETURN n", parameters={})


async def test_query_knowledge_graph_empty_result():
    mock_driver, _ = _make_mock_driver([])

    result = await query_knowledge_graph(
        "MATCH (n) RETURN n LIMIT 0", driver=mock_driver
    )

    assert result == []


async def test_query_knowledge_graph_with_parameters():
    mock_driver, mock_session = _make_mock_driver([{"name": "Einstein"}])

    params = {"entity_id": "e1", "limit": 5}
    result = await query_knowledge_graph(
        "MATCH (e:Entity {id: $entity_id}) RETURN e LIMIT $limit",
        driver=mock_driver,
        parameters=params,
    )

    assert result == [{"name": "Einstein"}]
    mock_session.run.assert_called_once_with(
        "MATCH (e:Entity {id: $entity_id}) RETURN e LIMIT $limit",
        parameters=params,
    )


async def test_query_knowledge_graph_none_parameters_uses_empty_dict():
    mock_driver, mock_session = _make_mock_driver([])

    await query_knowledge_graph(
        "MATCH (n) RETURN n", driver=mock_driver, parameters=None
    )

    mock_session.run.assert_called_once_with("MATCH (n) RETURN n", parameters={})
