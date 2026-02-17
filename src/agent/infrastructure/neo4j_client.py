# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Neo4j async driver manager with dependency injection.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Neo4j async driver manager with dependency injection."""

import asyncio

from neo4j import AsyncDriver, AsyncGraphDatabase


class Neo4jClient:
    """Manager for Neo4j asynchronous driver with per-loop caching.

    Avoids 'Future attached to a different loop' errors when LangGraph
    runs nodes in parallel.
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        """Initialize with connection parameters."""
        self._uri = uri
        self._username = username
        self._password = password
        self._drivers: dict[int, AsyncDriver] = {}

    async def get_driver(self) -> AsyncDriver:
        """Get or lazily initialize the Neo4j async driver for current loop.

        Returns:
            AsyncDriver: The Neo4j async driver instance.
        """
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
        if loop_id not in self._drivers:
            self._drivers[loop_id] = await asyncio.to_thread(
                AsyncGraphDatabase.driver,
                self._uri,
                auth=(self._username, self._password),
                max_connection_lifetime=300,
            )
        return self._drivers[loop_id]

    async def close(self) -> None:
        """Clean up all driver connections across all loops."""
        for driver in self._drivers.values():
            await driver.close()
        self._drivers.clear()
