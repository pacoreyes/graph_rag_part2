"""Tests for app.py — Chainlit entry point wired to the LangGraph agent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app as app_module
from app import _extract_ai_response

# ---------------------------------------------------------------------------
# _extract_ai_response — pure function tests
# ---------------------------------------------------------------------------


class TestExtractAiResponse:
    def test_last_ai_message_returned(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="hello back"),
        ]
        assert _extract_ai_response(messages) == "hello back"

    def test_multiple_ai_messages_returns_last(self):
        messages = [
            AIMessage(content="first"),
            HumanMessage(content="follow-up"),
            AIMessage(content="second"),
        ]
        assert _extract_ai_response(messages) == "second"

    def test_no_ai_message_returns_fallback(self):
        messages = [HumanMessage(content="hello")]
        assert _extract_ai_response(messages) == "No response generated."

    def test_empty_list_returns_fallback(self):
        assert _extract_ai_response([]) == "No response generated."


# ---------------------------------------------------------------------------
# on_chat_start
# ---------------------------------------------------------------------------


class TestOnChatStart:
    @pytest.mark.anyio
    async def test_session_initialized_with_empty_history(self, monkeypatch):
        mock_session = MagicMock()
        mock_cl = SimpleNamespace(user_session=mock_session)
        monkeypatch.setattr(app_module, "cl", mock_cl)

        await app_module.on_chat_start()

        mock_session.set.assert_called_once_with("history", [])


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


class TestOnMessage:
    @pytest.mark.anyio
    async def test_invokes_graph_and_sends_response(self, monkeypatch):
        ai_reply = AIMessage(content="I am the agent")
        graph_result = {
            "messages": [HumanMessage(content="hi"), ai_reply],
        }

        mock_session = MagicMock()
        mock_session.get.return_value = []  # empty history

        mock_send = AsyncMock()
        mock_message_instance = MagicMock()
        mock_message_instance.send = mock_send
        mock_message_cls = MagicMock(return_value=mock_message_instance)

        mock_cl = SimpleNamespace(
            user_session=mock_session,
            Message=mock_message_cls,
        )
        monkeypatch.setattr(app_module, "cl", mock_cl)

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=graph_result)
        monkeypatch.setattr(app_module, "graph", mock_graph)

        user_msg = MagicMock()
        user_msg.content = "hi"
        await app_module.on_message(user_msg)

        # Graph was invoked with the user message
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert len(call_args["messages"]) == 1
        assert isinstance(call_args["messages"][0], HumanMessage)
        assert call_args["messages"][0].content == "hi"

        # Chainlit message sent with AI reply
        mock_message_cls.assert_called_once_with(content="I am the agent")
        mock_send.assert_awaited_once()

        # History updated to graph output
        mock_session.set.assert_called_with(
            "history",
            list(graph_result["messages"]),
        )

    @pytest.mark.anyio
    async def test_fallback_when_graph_returns_no_ai_message(self, monkeypatch):
        graph_result = {"messages": [HumanMessage(content="hi")]}

        mock_session = MagicMock()
        mock_session.get.return_value = []

        mock_send = AsyncMock()
        mock_message_instance = MagicMock()
        mock_message_instance.send = mock_send
        mock_message_cls = MagicMock(return_value=mock_message_instance)

        mock_cl = SimpleNamespace(
            user_session=mock_session,
            Message=mock_message_cls,
        )
        monkeypatch.setattr(app_module, "cl", mock_cl)

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=graph_result)
        monkeypatch.setattr(app_module, "graph", mock_graph)

        user_msg = MagicMock()
        user_msg.content = "hi"
        await app_module.on_message(user_msg)

        mock_message_cls.assert_called_once_with(
            content="No response generated.",
        )
