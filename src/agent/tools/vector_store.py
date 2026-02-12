"""Reusable Pinecone vector search function.

Pure function with dependency injection — no global config or singletons.
"""

from pinecone import Pinecone


def vector_search(
    query_vector: list[float],
    client: Pinecone,
    index_name: str,
    top_k: int = 5,
    filter_dict: dict | None = None,
) -> dict:
    """Search a Pinecone vector index.

    Args:
        query_vector: The query embedding vector.
        client: Pinecone client instance (injected).
        index_name: Name of the Pinecone index to query.
        top_k: Number of top results to return.
        filter_dict: Optional metadata filter for the query.

    Returns:
        dict: Pinecone query response with matches and metadata.
    """
    index = client.Index(index_name)
    kwargs: dict = dict(vector=query_vector, top_k=top_k, include_metadata=True)
    if filter_dict is not None:
        kwargs["filter"] = filter_dict
    return index.query(**kwargs)
