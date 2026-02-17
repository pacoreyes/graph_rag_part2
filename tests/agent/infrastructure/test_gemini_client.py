# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from unittest.mock import MagicMock, patch

import pytest
from agent.infrastructure.gemini_client import GeminiClient

pytestmark = pytest.mark.anyio


def test_gemini_client_stores_api_key():
    client = GeminiClient(api_key="test-key", schema_path="dummy/path.json")
    assert client._api_key == "test-key"
    assert str(client._schema_path) == "dummy/path.json"


@patch("agent.infrastructure.gemini_client.genai.Client")
async def test_get_client_creates_client_once(mock_client_cls):
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance
    
    client = GeminiClient(api_key="test-key", schema_path="dummy/path.json")
    
    # First call initializes
    result1 = await client.get_client()
    assert result1 == mock_instance
    mock_client_cls.assert_called_once_with(api_key="test-key")
    
    # Second call returns cached
    result2 = await client.get_client()
    assert result2 == mock_instance
    mock_client_cls.assert_called_once() # Call count remains 1


@patch("agent.infrastructure.gemini_client.genai.Client")
async def test_get_client_returns_genai_client(mock_client_cls):
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance
    
    client = GeminiClient(api_key="test-key", schema_path="dummy/path.json")
    result = await client.get_client()
    
    assert result == mock_instance
