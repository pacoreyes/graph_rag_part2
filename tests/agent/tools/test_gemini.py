from unittest.mock import MagicMock

from agent.tools.gemini import gemini_embed, gemini_generate


def test_gemini_embed_returns_vector():
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]

    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_response

    result = gemini_embed(client=mock_client, text="hello world")

    assert result == [0.1, 0.2, 0.3]
    mock_client.models.embed_content.assert_called_once_with(
        model="gemini-embedding-001", contents="hello world"
    )


def test_gemini_embed_custom_model():
    mock_embedding = MagicMock()
    mock_embedding.values = [0.5]

    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_response

    gemini_embed(client=mock_client, text="test", model="custom-embed-model")

    mock_client.models.embed_content.assert_called_once_with(
        model="custom-embed-model", contents="test"
    )


def test_gemini_generate_returns_text():
    mock_response = MagicMock()
    mock_response.text = "Generated response"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    result = gemini_generate(client=mock_client, prompt="Say hello")

    assert result == "Generated response"
    mock_client.models.generate_content.assert_called_once_with(
        model="models/gemini-2.0-flash", contents="Say hello"
    )


def test_gemini_generate_custom_model():
    mock_response = MagicMock()
    mock_response.text = "Custom output"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    gemini_generate(client=mock_client, prompt="test", model="custom-gen-model")

    mock_client.models.generate_content.assert_called_once_with(
        model="custom-gen-model", contents="test"
    )
