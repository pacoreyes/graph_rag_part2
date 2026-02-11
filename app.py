"""Chainlit chat UI wired to the LangGraph agent.

Entry point for ``chainlit run app.py``.
Maintains per-session conversation history and delegates to the compiled
LangGraph graph for every user message.
"""

import chainlit as cl
import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.graph import graph

logger = structlog.get_logger(__name__)


def _extract_ai_response(messages: list[BaseMessage]) -> str:
    """Return the content of the last AIMessage in *messages*.

    Args:
        messages: Sequence of LangChain messages returned by the graph.

    Returns:
        str: The text content of the last AIMessage, or a fallback string
            if no AIMessage is found.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return "No response generated."


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize an empty message history for the new session."""
    cl.user_session.set("history", [])
    logger.info("chat_session_started")


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle an incoming user message.

    Appends the message to session history, invokes the LangGraph graph,
    extracts the assistant reply, and sends it back through Chainlit.

    Args:
        message: The Chainlit message object from the user.
    """
    history: list[BaseMessage] = cl.user_session.get("history")  # type: ignore[assignment]

    user_msg = HumanMessage(content=message.content)
    history.append(user_msg)

    logger.info("invoking_graph", num_messages=len(history))
    result = await graph.ainvoke({"messages": history})

    response_messages: list[BaseMessage] = result.get("messages", [])
    reply_text = _extract_ai_response(response_messages)

    # Keep history in sync with what the graph returned
    cl.user_session.set("history", list(response_messages))

    await cl.Message(content=reply_text).send()
    logger.info("response_sent", reply_length=len(reply_text))
