# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for Google Gemini tools.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for Google Gemini tools."""

from unittest.mock import MagicMock
from agent.tools.gemini import gemini_generate

def test_gemini_generate_returns_text():
    """Verify that gemini_generate returns text content correctly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "generated response"
    mock_client.models.generate_content.return_value = mock_response
    
    result = gemini_generate(client=mock_client, prompt="hello")
    
    assert result == "generated response"
    mock_client.models.generate_content.assert_called_once()
