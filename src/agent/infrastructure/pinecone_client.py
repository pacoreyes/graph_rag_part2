# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Pinecone client manager with dependency injection.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import asyncio
from pinecone import Pinecone


class PineconeClient:
    """Manager for Pinecone client.

    Args:
        api_key: Pinecone API key.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize with API key."""
        self._api_key = api_key
        self._client: Pinecone | None = None

    async def get_client(self) -> Pinecone:
        """Get or lazily initialize the Pinecone client.

        Returns:
            Pinecone: The Pinecone client instance.
        """
        if self._client is None:
            self._client = await asyncio.to_thread(Pinecone, api_key=self._api_key)
        return self._client
