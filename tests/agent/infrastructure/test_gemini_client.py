from unittest.mock import MagicMock, patch

from agent.infrastructure.gemini_client import GeminiClient


def test_gemini_client_stores_api_key():
    client = GeminiClient(api_key="test-key")
    assert client._api_key == "test-key"


@patch("agent.infrastructure.gemini_client.genai.Client")
def test_get_client_creates_client_once(mock_client_cls):
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    client = GeminiClient(api_key="test-key")
    c1 = client.get_client()
    c2 = client.get_client()

    assert c1 is c2
    mock_client_cls.assert_called_once_with(api_key="test-key")


@patch("agent.infrastructure.gemini_client.genai.Client")
def test_get_client_returns_genai_client(mock_client_cls):
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    client = GeminiClient(api_key="test-key")
    result = client.get_client()

    assert result is mock_instance
