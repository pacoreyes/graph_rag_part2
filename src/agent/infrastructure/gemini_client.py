"""Google Gemini client manager with dependency injection."""

import asyncio
from google import genai


class GeminiClient:
    """Manager for Google Gemini client.

    Args:
        api_key: Gemini API key.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize with API key."""
        self._api_key = api_key
        self._client: genai.Client | None = None

    async def get_client(self) -> genai.Client:
        """Get or lazily initialize the Gemini client.

        Returns:
            genai.Client: The Gemini client instance.
        """
        if self._client is None:
            self._client = await asyncio.to_thread(genai.Client, api_key=self._api_key)
        return self._client
