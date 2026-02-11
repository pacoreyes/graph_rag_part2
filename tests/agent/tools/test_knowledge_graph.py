from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.knowledge_graph import query_knowledge_graph

pytestmark = pytest.mark.anyio


async def test_query_knowledge_graph_returns_data():
    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[{"n": "node1"}, {"n": "node2"}])

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_ctx

    result = await query_knowledge_graph("MATCH (n) RETURN n", driver=mock_driver)

    assert result == [{"n": "node1"}, {"n": "node2"}]
    mock_session.run.assert_called_once_with("MATCH (n) RETURN n")


async def test_query_knowledge_graph_empty_result():
    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[])

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_ctx

    result = await query_knowledge_graph(
        "MATCH (n) RETURN n LIMIT 0", driver=mock_driver
    )

    assert result == []
