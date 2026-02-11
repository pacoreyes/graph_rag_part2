"""Pinecone client manager with dependency injection."""

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

    def get_client(self) -> Pinecone:
        """Get or lazily initialize the Pinecone client.

        Returns:
            Pinecone: The Pinecone client instance.
        """
        if self._client is None:
            self._client = Pinecone(api_key=self._api_key)
        return self._client
