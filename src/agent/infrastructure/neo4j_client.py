# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Neo4j async driver manager with dependency injection.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import asyncio
import weakref

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase

logger = structlog.get_logger()


class Neo4jClient:
    """Manager for Neo4j asynchronous driver with per-loop caching.

    Caches one driver per event loop.  When a loop is garbage-collected
    its driver is closed automatically via a weak-reference finalizer,
    preventing leaked connections.
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        """Initialize with connection parameters.

        Args:
            uri: Neo4j connection URI.
            username: Neo4j username.
            password: Neo4j password.
        """
        self._uri = uri
        self._username = username
        self._password = password
        self._drivers: dict[int, AsyncDriver] = {}
        self._loop_refs: dict[int, weakref.ref] = {}

    async def get_driver(self) -> AsyncDriver:
        """Get or lazily initialize the Neo4j async driver for the current loop.

        Returns:
            AsyncDriver: The Neo4j async driver instance.
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        # Check if cached driver's loop is still alive
        if loop_id in self._drivers:
            ref = self._loop_refs.get(loop_id)
            if ref is not None and ref() is not None:
                return self._drivers[loop_id]
            # Stale entry — loop was recycled with same id
            await self._close_driver(loop_id)

        driver = await asyncio.to_thread(
            AsyncGraphDatabase.driver,
            self._uri,
            auth=(self._username, self._password),
            max_connection_lifetime=300,
        )
        self._drivers[loop_id] = driver
        self._loop_refs[loop_id] = weakref.ref(
            loop, lambda _ref, _lid=loop_id: self._on_loop_gc(_lid)
        )
        return driver

    def _on_loop_gc(self, loop_id: int) -> None:
        """Weak-reference callback: schedule driver cleanup when a loop is GC'd."""
        driver = self._drivers.pop(loop_id, None)
        self._loop_refs.pop(loop_id, None)
        if driver is not None:
            logger.info("neo4j_driver_gc_cleanup", loop_id=loop_id)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(driver.close())
            except RuntimeError:
                pass

    async def _close_driver(self, loop_id: int) -> None:
        """Close and remove a cached driver by loop id."""
        driver = self._drivers.pop(loop_id, None)
        self._loop_refs.pop(loop_id, None)
        if driver is not None:
            await driver.close()

    async def close(self) -> None:
        """Clean up all driver connections across all loops."""
        for driver in self._drivers.values():
            await driver.close()
        self._drivers.clear()
        self._loop_refs.clear()
