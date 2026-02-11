"""Reusable Pinecone vector search function.

Pure function with dependency injection — no global config or singletons.
"""

from pinecone import Pinecone


def vector_search(
    query_vector: list[float],
    client: Pinecone,
    index_name: str,
    top_k: int = 5,
) -> dict:
    """Search a Pinecone vector index.

    Args:
        query_vector: The query embedding vector.
        client: Pinecone client instance (injected).
        index_name: Name of the Pinecone index to query.
        top_k: Number of top results to return.

    Returns:
        dict: Pinecone query response with matches and metadata.
    """
    index = client.Index(index_name)
    return index.query(vector=query_vector, top_k=top_k, include_metadata=True)
