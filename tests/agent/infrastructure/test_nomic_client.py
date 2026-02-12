from unittest.mock import MagicMock, patch

from agent.infrastructure.nomic_client import NomicClient


@patch("agent.infrastructure.nomic_client.AutoModel")
def test_get_model_lazy_init(mock_auto_model):
    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    client = NomicClient(model_name="test-model")
    result = client.get_model()

    mock_auto_model.from_pretrained.assert_called_once_with(
        "test-model", trust_remote_code=True
    )
    mock_model.eval.assert_called_once()
    assert result is mock_model


@patch("agent.infrastructure.nomic_client.AutoModel")
def test_get_model_singleton(mock_auto_model):
    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    client = NomicClient(model_name="test-model")
    result1 = client.get_model()
    result2 = client.get_model()

    assert result1 is result2
    assert mock_auto_model.from_pretrained.call_count == 1


@patch("agent.infrastructure.nomic_client.AutoTokenizer")
def test_get_tokenizer_lazy_init(mock_auto_tokenizer):
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    client = NomicClient(model_name="test-model")
    result = client.get_tokenizer()

    mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model")
    assert result is mock_tokenizer


@patch("agent.infrastructure.nomic_client.AutoTokenizer")
def test_get_tokenizer_singleton(mock_auto_tokenizer):
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    client = NomicClient(model_name="test-model")
    result1 = client.get_tokenizer()
    result2 = client.get_tokenizer()

    assert result1 is result2
    assert mock_auto_tokenizer.from_pretrained.call_count == 1
