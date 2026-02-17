# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Pydantic models for constrained LLM generation.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Pydantic models for constrained LLM generation."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class RouterResponse(BaseModel):
    """Schema for the Intelligent Router node."""
    strategy: str = Field(description="One of: local, global, drift, structural, hybrid")
    is_fast_track: bool = Field(description="True if the query is a simple single-entity factoid lookup.")
    target_entity_types: List[str] = Field(description="Relevant entity types from schema to filter search.")
    plan: str = Field(description="Short reasoning for the chosen strategy.")


class CypherResponse(BaseModel):
    """Schema for the Structural Expert (NL-to-Cypher) node."""
    cypher: str = Field(description="A single valid Neo4j Cypher query. Do NOT use EXPLAIN.")


class EvidenceItem(BaseModel):
    """A single piece of evidence tied to a citation index."""
    index: int = Field(description="The numeric index corresponding to the Atomic Knowledge Unit.")
    content: str = Field(description="Verbatim excerpt or fact summary proving the claim.")


class SynthesisResponse(BaseModel):
    """Schema for the Answer Synthesizer node."""
    answer: str = Field(description="The final narrative answer with [N] citations.")
    evidence: List[EvidenceItem] = Field(description="List of evidence items supporting the answer.")
