# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Reusable Pinecone vector search and inference functions.
# Pure functions with dependency injection — no global config or singletons.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from pinecone import Pinecone


def pinecone_embed(
    client: Pinecone,
    text: str,
    model: str = "llama-text-embed-v2",
) -> list[float]:
    """Generate an embedding vector using Pinecone Inference API.

    Args:
        client: Pinecone client instance (injected).
        text: Text to embed.
        model: Pinecone inference model name.

    Returns:
        list[float]: The embedding vector (1024-dim for llama-text-embed-v2).
    """
    response = client.inference.embed(
        model=model,
        inputs=[text],
        parameters={"input_type": "query"},
    )
    return response.data[0].values


def vector_search(
    client: Pinecone,
    index_name: str,
    query_vector: list[float] | None = None,
    query_text: str | None = None,
    top_k: int = 5,
    filter_dict: dict | None = None,
) -> dict:
    """Search a Pinecone vector index.

    Supports both manual vector search and integrated inference (raw text).

    Args:
        client: Pinecone client instance (injected).
        index_name: Name of the Pinecone index to query.
        query_vector: The query embedding vector (optional if query_text is provided).
        query_text: Raw text for integrated inference (optional if query_vector is provided).
        top_k: Number of top results to return.
        filter_dict: Optional metadata filter for the query.

    Returns:
        dict: Pinecone query response with matches and metadata.

    Raises:
        ValueError: If neither query_vector nor query_text is provided.
    """
    if query_vector is None and query_text is None:
        raise ValueError("Either query_vector or query_text must be provided.")

    index = client.Index(index_name)
    kwargs: dict = dict(top_k=top_k, include_metadata=True)

    if query_text is not None:
        # Integrated Inference: pass raw text in 'data' parameter
        kwargs["data"] = query_text
    else:
        # Manual vector search: pass pre-computed embedding
        kwargs["vector"] = query_vector

    if filter_dict is not None:
        kwargs["filter"] = filter_dict
    return index.query(**kwargs)
