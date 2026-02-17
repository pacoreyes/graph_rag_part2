# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Runtime configuration for the LangGraph agent.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Runtime configuration for the LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Configuration:
    """Runtime configuration for the agent, decoupled from environment secrets."""

    model: str = field(
        default="gemini-2.0-flash",
        metadata={"description": "The name of the language model to use."},
    )
    retrieval_k: int = field(
        default=5,
        metadata={"description": "Number of documents to retrieve."},
    )
    community_level: int = field(
        default=2,
        metadata={
            "description": "Leiden community level for global search (0, 1, 2). Use -1 for all levels."
        },
    )
    neighborhood_depth: int = field(
        default=1,
        metadata={"description": "Number of hops for local expansion."},
    )
    similarity_threshold: float = field(
        default=0.5,
        metadata={"description": "Min cosine similarity for relation traversal."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: dict[str, Any] | None = None
    ) -> Configuration:
        """Extract configuration from LangGraph runtime config."""
        if not config or "configurable" not in config:
            return cls()
        configurable = config["configurable"]
        return cls(
            **{k: v for k, v in configurable.items() if k in cls.__dataclass_fields__}
        )
