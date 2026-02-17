# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Graph state definition for the LangGraph agent.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class State:
    """SOTA Graph State using Annotated and add_messages for history management."""

    messages: Annotated[Sequence[BaseMessage], add_messages] = field(
        default_factory=list
    )
    # Strategy selected by router
    strategy: str = ""

    # Reasoning
    plan: str = ""
    is_fast_track: bool = False

    # Generated Cypher query for structural strategy
    generated_cypher: str = ""
    cypher_result: list[dict] = field(default_factory=list)
    cypher_error: str = ""

    # Entity types identified by query_analyzer for surgical filtering
    target_entity_types: list[str] = field(default_factory=list)

    # Snowflake arctic-embed-s embedding of the user query (384-dim)
    query_embedding: list[float] = field(default_factory=list)

    # Retrieval results (additive reducers for parallel/sequential merging)
    entities: Annotated[list[dict], operator.add] = field(default_factory=list)
    relationships: Annotated[list[dict], operator.add] = field(default_factory=list)
    community_reports: Annotated[list[dict], operator.add] = field(default_factory=list)
    chunk_evidence: Annotated[list[dict], operator.add] = field(default_factory=list)

    # Homogenized Atomic Knowledge Units for USA Framework
    akus: list[dict] = field(default_factory=list)

    # Source tracking
    source_qids: Annotated[list[str], operator.add] = field(default_factory=list)
    source_urls: dict[str, dict[str, str]] = field(default_factory=dict)
