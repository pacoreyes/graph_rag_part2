"""Reusable Neo4j knowledge graph query function and schema utilities.

Pure functions with dependency injection — no global config or singletons.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from neo4j import AsyncDriver


def sanitize_lucene_query(query: str) -> str:
    """Escape Lucene special characters in a search string.

    Special characters include: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /

    Args:
        query: The raw query string from the user.

    Returns:
        str: Sanitized query string safe for db.index.fulltext.queryNodes.
    """
    if not isinstance(query, str):
        query = str(query)
    # Escaping special Lucene characters with backslash
    # Note: We escape the backslash itself first
    special_chars = r'[+\-&|!(){}\[\]^"~*?:\\]'
    sanitized = re.sub(special_chars, r'\\\g<0>', query)

    # Handle logical operators if they are literal words
    sanitized = sanitized.replace(" AND ", " \\AND ").replace(" OR ", " \\OR ").replace(" NOT ", " \\NOT ")

    return sanitized.strip()


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


@lru_cache
def get_graph_schema(schema_path: Path) -> dict:
    """Load the machine-readable graph schema from assets.

    Args:
        schema_path: Path to the graph_schema.json file.

    Returns:
        dict: The graph schema containing node_labels, relationship_types, etc.
    """
    if not schema_path.exists():
        return {}
    with open(schema_path, "r") as f:
        return json.load(f)


def format_schema_for_prompt(schema_path: Path) -> str:
    """Format the schema into a concise string for LLM prompts.

    Args:
        schema_path: Path to the graph_schema.json file.

    Returns:
        str: Formatted schema description.
    """
    schema = get_graph_schema(schema_path)
    if not schema:
        return "Schema unavailable."

    # Exclude Community/IN_COMMUNITY as per TO_LANGGRAPH.md if they exist in schema
    node_labels = [
        label for label in schema.get("node_labels", []) if label != "Community"
    ]
    rel_types = [
        rel for rel in schema.get("relationship_types", []) if rel != "IN_COMMUNITY"
    ]

    lines = [
        "## Graph Schema",
        f"Node Types: {', '.join(node_labels)}",
        f"Relationship Types: {', '.join(rel_types)}",
    ]
    return "\n".join(lines)
