"""Reusable Neo4j knowledge graph query function.

Pure function with dependency injection — no global config or singletons.
"""

from neo4j import AsyncDriver


async def query_knowledge_graph(
    query: str,
    driver: AsyncDriver,
    parameters: dict | None = None,
) -> list[dict]:
    """Execute a Cypher query against the Neo4j knowledge graph.

    Args:
        query: Cypher query string to execute.
        driver: Neo4j async driver instance (injected).
        parameters: Optional Cypher query parameters for safe parameterized queries.

    Returns:
        list[dict]: List of result records as dictionaries.
    """
    async with driver.session() as session:
        result = await session.run(query, parameters=parameters or {})
        return await result.data()
