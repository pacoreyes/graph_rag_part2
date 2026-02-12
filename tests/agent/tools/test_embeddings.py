from unittest.mock import MagicMock

import torch

from agent.tools.embeddings import nomic_embed


def _make_mock_model_and_tokenizer(hidden_size=768):
    """Create mock model and tokenizer that return deterministic tensors."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }

    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(1, 5, hidden_size)

    mock_model = MagicMock()
    mock_model.return_value = mock_output

    return mock_model, mock_tokenizer


def test_nomic_embed_returns_list_of_floats():
    model, tokenizer = _make_mock_model_and_tokenizer()

    result = nomic_embed(model, tokenizer, "test query")

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_nomic_embed_applies_prefix():
    model, tokenizer = _make_mock_model_and_tokenizer()

    nomic_embed(model, tokenizer, "hello", prefix="search_query: ")

    tokenizer.assert_called_once()
    call_args = tokenizer.call_args[0][0]
    assert call_args == "search_query: hello"


def test_nomic_embed_custom_prefix():
    model, tokenizer = _make_mock_model_and_tokenizer()

    nomic_embed(model, tokenizer, "hello", prefix="search_document: ")

    call_args = tokenizer.call_args[0][0]
    assert call_args == "search_document: hello"


def test_nomic_embed_output_is_normalized():
    model, tokenizer = _make_mock_model_and_tokenizer()

    result = nomic_embed(model, tokenizer, "test")

    # L2 norm should be ~1.0
    norm = sum(v**2 for v in result) ** 0.5
    assert abs(norm - 1.0) < 1e-5
