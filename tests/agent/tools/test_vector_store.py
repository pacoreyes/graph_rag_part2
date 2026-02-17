# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# No description available.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from unittest.mock import MagicMock

from agent.tools.vector_store import vector_search


def test_vector_search_calls_index_query():
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [{"id": "1", "score": 0.9, "metadata": {"text": "hello"}}]
    }

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    result = vector_search(
        query_vector=[0.1, 0.2, 0.3],
        client=mock_client,
        index_name="test-index",
        top_k=3,
    )

    mock_client.Index.assert_called_once_with("test-index")
    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3], top_k=3, include_metadata=True
    )
    assert "matches" in result


def test_vector_search_default_top_k():
    mock_index = MagicMock()
    mock_index.query.return_value = {"matches": []}

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    vector_search(
        query_vector=[0.1],
        client=mock_client,
        index_name="test-index",
    )

    mock_index.query.assert_called_once_with(
        vector=[0.1], top_k=5, include_metadata=True
    )


def test_vector_search_with_filter_dict():
    mock_index = MagicMock()
    mock_index.query.return_value = {"matches": []}

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    filter_dict = {"article_qid": {"$in": ["Q1", "Q2"]}}
    vector_search(
        query_vector=[0.1],
        client=mock_client,
        index_name="test-index",
        filter_dict=filter_dict,
    )

    mock_index.query.assert_called_once_with(
        vector=[0.1], top_k=5, include_metadata=True, filter=filter_dict
    )


def test_vector_search_none_filter_not_passed():
    mock_index = MagicMock()
    mock_index.query.return_value = {"matches": []}

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    vector_search(
        query_vector=[0.1],
        client=mock_client,
        index_name="test-index",
        filter_dict=None,
    )

    # filter kwarg should NOT be in the call
    call_kwargs = mock_index.query.call_args[1]
    assert "filter" not in call_kwargs
