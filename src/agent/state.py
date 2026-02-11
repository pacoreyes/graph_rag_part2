"""Graph state definition for the LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class State:
    """SOTA Graph State using Annotated and add_messages for history management."""
    messages: Annotated[Sequence[BaseMessage], add_messages] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
