from unittest.mock import MagicMock, patch

from agent.infrastructure.pinecone_client import PineconeClient


def test_pinecone_client_stores_api_key():
    client = PineconeClient(api_key="test-key")
    assert client._api_key == "test-key"


@patch("agent.infrastructure.pinecone_client.Pinecone")
def test_get_client_creates_client_once(mock_pinecone_cls):
    mock_instance = MagicMock()
    mock_pinecone_cls.return_value = mock_instance

    client = PineconeClient(api_key="test-key")
    pc1 = client.get_client()
    pc2 = client.get_client()

    assert pc1 is pc2
    mock_pinecone_cls.assert_called_once_with(api_key="test-key")


@patch("agent.infrastructure.pinecone_client.Pinecone")
def test_get_client_returns_pinecone_instance(mock_pinecone_cls):
    mock_instance = MagicMock()
    mock_pinecone_cls.return_value = mock_instance

    client = PineconeClient(api_key="test-key")
    result = client.get_client()

    assert result is mock_instance
