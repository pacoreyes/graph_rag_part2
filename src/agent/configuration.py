"""Runtime configuration for the LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Configuration:
    """Runtime configuration for the agent, decoupled from environment secrets."""
    model: str = field(
        default="gemini-1.5-flash",
        metadata={"description": "The name of the language model to use."},
    )
    retrieval_k: int = field(
        default=5,
        metadata={"description": "Number of documents to retrieve."},
    )

    @classmethod
    def from_runnable_config(cls, config: dict[str, Any] | None = None) -> Configuration:
        """Extract configuration from LangGraph runtime config."""
        if not config or "configurable" not in config:
            return cls()
        configurable = config["configurable"]
        return cls(**{k: v for k, v in configurable.items() if k in cls.__dataclass_fields__})
