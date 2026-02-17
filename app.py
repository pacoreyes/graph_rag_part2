# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Chainlit chat UI wired to the LangGraph agent.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

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

    Appends the message to session history, invokes the LangGraph graph using
    streaming to show intermediate steps in the UI, extracts the final reply,
    and updates history.

    Args:
        message: The Chainlit message object from the user.
    """
    history: list[BaseMessage] = cl.user_session.get("history")

    user_msg = HumanMessage(content=message.content)
    history.append(user_msg)

    logger.info("invoking_graph_stream", num_messages=len(history))
    
    # Track the full history to preserve conversation context
    full_history = list(history)
    final_reply = "No response generated."

    # Create a single persistent step that updates in-place
    active_step = cl.Step(name="Agent Starting...", type="run")
    await active_step.send()

    # Stream graph updates to show steps in Chainlit
    async for output in graph.astream({"messages": history}, stream_mode="updates"):
        for node_name, state_update in output.items():
            if state_update is None:
                continue

            # Accumulate messages as they are generated
            new_messages = state_update.get("messages", [])
            full_history.extend(new_messages)
            
            # Check for the final answer in the last node's output
            if node_name == "synthesize_answer":
                final_reply = _extract_ai_response(new_messages)

            # Update the same step instead of creating new ones
            active_step.name = f"{node_name}"

            if node_name == "router":
                strategy = state_update.get("strategy", "unknown")
                active_step.output = f"Strategy: {strategy}. Plan: {state_update.get('plan', '')}"
            elif node_name == "embed_query":
                active_step.output = "Query embedded."
            elif node_name == "entity_search":
                count = len(state_update.get("entities", []))
                active_step.output = f"Retrieved {count} entities from Knowledge Graph."
            elif node_name == "neighborhood_expand":
                count = len(state_update.get("relationships", []))
                active_step.output = f"Expanded neighborhood: {count} relationships found."
            elif node_name == "community_search":
                count = len(state_update.get("community_reports", []))
                active_step.output = f"Retrieved {count} community summaries."
            elif node_name == "community_members_search":
                count = len(state_update.get("entities", []))
                active_step.output = f"Found {count} community members."
            elif node_name == "chunk_search":
                count = len(state_update.get("chunk_evidence", []))
                active_step.output = f"Retrieved {count} text evidence chunks."
            elif node_name == "nl_to_cypher":
                if state_update.get("cypher_error"):
                    active_step.output = f"Structural query failed: {state_update.get('cypher_error')}"
                else:
                    count = len(state_update.get("cypher_result", []))
                    active_step.output = f"Structural query found {count} results."
            elif node_name == "resolve_sources":
                count = len(state_update.get("source_urls", {}))
                active_step.output = f"Resolved {count} Wikipedia sources."
            elif node_name == "synthesize_answer":
                active_step.output = "Generating answer with citations."
            else:
                active_step.output = f"Step {node_name} completed."
            
            await active_step.update()

    active_step.name = "Retrieval and Analysis Complete"
    await active_step.update()

    # Keep history in sync with the full sequence of messages
    cl.user_session.set("history", full_history)

    await cl.Message(content=final_reply, author="Assistant").send()
    logger.info("response_sent", reply_length=len(final_reply))
