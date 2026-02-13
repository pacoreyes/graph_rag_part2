"""Graph state definition for the LangGraph agent."""

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
    retrieved_context: list[str] = field(default_factory=list)

    # Strategy selected by query_analyzer
    strategy: str = ""

    # Reasoning and Loop management
    plan: str = ""
    critique: str = ""
    iteration_count: int = 0
    
    # Evaluator output
    retrieval_guide: str = ""
    skip_deep_search: bool = False

    # Generated Cypher query for structural strategy
    generated_cypher: str = ""
    cypher_result: list[dict] = field(default_factory=list)
    cypher_error: str = ""

    # Entity types identified by query_analyzer for surgical filtering
    target_entity_types: list[str] = field(default_factory=list)

    # Nomic embedding of the user query (768-dim)
    query_embedding: list[float] = field(default_factory=list)

    # Retrieval results (additive reducers for parallel/sequential merging)
    entities: Annotated[list[dict], operator.add] = field(default_factory=list)
    relationships: Annotated[list[dict], operator.add] = field(default_factory=list)
    community_reports: Annotated[list[dict], operator.add] = field(default_factory=list)
    chunk_evidence: Annotated[list[dict], operator.add] = field(default_factory=list)

    # Source tracking
    source_qids: Annotated[list[str], operator.add] = field(default_factory=list)
    source_urls: dict[str, dict[str, str]] = field(default_factory=dict)
