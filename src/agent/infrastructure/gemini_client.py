"""Google Gemini client manager with dependency injection."""

import asyncio
from google import genai


from pathlib import Path
from agent.settings import settings
from agent.tools.knowledge_graph import format_schema_for_prompt

class GeminiClient:
    """Manager for Google Gemini client.

    Args:
        api_key: Gemini API key.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize with API key."""
        self._api_key = api_key
        self._client: genai.Client | None = None
        self._schema_instruction: str | None = None

    async def get_client(self) -> genai.Client:
        """Get or lazily initialize the Gemini client."""
        if self._client is None:
            self._client = await asyncio.to_thread(genai.Client, api_key=self._api_key)
        return self._client

    def get_schema_instruction(self) -> str:
        """Get the cached graph schema as a system instruction."""
        if self._schema_instruction is None:
            schema_path = Path(settings.data_volume_path) / "graph_schema.json"
            self._schema_instruction = format_schema_for_prompt(schema_path)
        return self._schema_instruction
