# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for the HFEmbeddingClient.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for the HFEmbeddingClient."""

from unittest.mock import patch
import pytest
from agent.infrastructure.hf_embedding_client import HFEmbeddingClient

pytestmark = pytest.mark.anyio

@patch("agent.infrastructure.hf_embedding_client.requests.post")
async def test_embed_success(mock_post):
    """Test successful embedding generation."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = [[0.1, 0.2, 0.3]]
    
    client = HFEmbeddingClient(model_name="test-model", api_token="test-token")
    result = await client.embed("test query")
    
    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once()
    # Check if prefix is added and it's a list
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["inputs"] == ["Represent this sentence for searching relevant passages: test query"]
    assert "pipeline/feature-extraction" in client._api_url

@patch("agent.infrastructure.hf_embedding_client.requests.post")
async def test_embed_error(mock_post):
    """Test error handling in embedding generation."""
    mock_post.return_value.status_code = 500
    mock_post.return_value.text = "Internal Server Error"
    
    client = HFEmbeddingClient(model_name="test-model", api_token="test-token")
    with pytest.raises(Exception):
        await client.embed("test query")
