# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for Google Gemini tools.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for Google Gemini tools."""

from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError

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


def test_gemini_generate_returns_parsed_json():
    """Verify JSON mode returns parsed response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = {"answer": "42"}
    mock_client.models.generate_content.return_value = mock_response

    result = gemini_generate(
        client=mock_client,
        prompt="hello",
        response_mime_type="application/json",
    )

    assert result == {"answer": "42"}


@patch("agent.tools.gemini.time.sleep")
def test_gemini_generate_retries_on_429(mock_sleep):
    """Verify exponential backoff with jitter on 429 errors."""
    mock_client = MagicMock()

    error_429 = ClientError(429, {})

    mock_response = MagicMock()
    mock_response.text = "success after retry"

    # Fail twice with 429, then succeed
    mock_client.models.generate_content.side_effect = [
        error_429,
        error_429,
        mock_response,
    ]

    result = gemini_generate(client=mock_client, prompt="hello")

    assert result == "success after retry"
    assert mock_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@patch("agent.tools.gemini.time.sleep")
def test_gemini_generate_raises_after_max_retries(mock_sleep):
    """Verify it raises after exhausting all retries."""
    mock_client = MagicMock()

    error_429 = ClientError(429, {})
    mock_client.models.generate_content.side_effect = error_429

    with pytest.raises(ClientError):
        gemini_generate(client=mock_client, prompt="hello")

    assert mock_client.models.generate_content.call_count == 5


def test_gemini_generate_raises_non_429_immediately():
    """Verify non-429 errors are raised without retry."""
    mock_client = MagicMock()

    error_400 = ClientError(400, {})
    mock_client.models.generate_content.side_effect = error_400

    with pytest.raises(ClientError):
        gemini_generate(client=mock_client, prompt="hello")

    assert mock_client.models.generate_content.call_count == 1
