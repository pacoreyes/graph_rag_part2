"""Neo4j async driver manager with dependency injection."""

from neo4j import AsyncDriver, AsyncGraphDatabase


class Neo4jClient:
    """Manager for Neo4j asynchronous driver.

    Args:
        uri: Neo4j connection URI.
        username: Neo4j username.
        password: Neo4j password.
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        """Initialize with connection parameters."""
        self._uri = uri
        self._username = username
        self._password = password
        self._driver: AsyncDriver | None = None

    def get_driver(self) -> AsyncDriver:
        """Get or lazily initialize the Neo4j async driver.

        Returns:
            AsyncDriver: The Neo4j async driver instance.
        """
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._username, self._password)
            )
        return self._driver

    async def close(self) -> None:
        """Clean up the driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
