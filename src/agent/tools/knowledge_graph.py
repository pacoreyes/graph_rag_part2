"""Reusable Neo4j knowledge graph query function.

Pure function with dependency injection — no global config or singletons.
"""

from neo4j import AsyncDriver


async def query_knowledge_graph(
    query: str, driver: AsyncDriver
) -> list[dict]:
    """Execute a Cypher query against the Neo4j knowledge graph.

    Args:
        query: Cypher query string to execute.
        driver: Neo4j async driver instance (injected).

    Returns:
        list[dict]: List of result records as dictionaries.
    """
    async with driver.session() as session:
        result = await session.run(query)
        return await result.data()
