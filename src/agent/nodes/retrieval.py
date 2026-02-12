"""Retrieval nodes for the LangGraph agent.

Each function is a LangGraph node that performs one step in
the multi-strategy retrieval pipeline (Local / Global / Hybrid).
"""

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import (
    neo4j_client,
    nomic_client,
    pinecone_client,
)
from agent.settings import settings
from agent.state import State
from agent.tools.embeddings import nomic_embed
from agent.tools.knowledge_graph import query_knowledge_graph
from agent.tools.source_resolver import resolve_source_urls
from agent.tools.vector_store import vector_search

logger = structlog.get_logger()

# --- Cypher queries (Rule 8: never modify or delete) ---

ENTITY_FULLTEXT_SEARCH = (
    "CALL db.index.fulltext.queryNodes('entityNameIndex', $query) "
    "YIELD node, score "
    "RETURN node.id AS id, node.name AS name, node.description AS description, "
    "node.qid AS qid, score "
    "ORDER BY score DESC LIMIT $limit"
)

ENTITY_NEIGHBORHOOD = (
    "MATCH (e:Entity {id: $entity_id})-[r]->(t:Entity) "
    "RETURN e.id AS source_id, e.name AS source_name, "
    "type(r) AS relationship, r.description AS rel_description, "
    "t.id AS target_id, t.name AS target_name, t.qid AS target_qid "
    "LIMIT $limit"
)

COMMUNITY_MEMBERS = (
    "MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community {id: $community_id}) "
    "RETURN e.id AS id, e.name AS name, e.description AS description, "
    "e.qid AS qid, e.pagerank AS pagerank "
    "ORDER BY e.pagerank DESC LIMIT 20"
)


async def embed_query(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Compute Nomic embedding for the user query.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with query_embedding.
    """
    _ = config
    last_message = state.messages[-1]
    query_text = last_message.content

    model = nomic_client.get_model()
    tokenizer = nomic_client.get_tokenizer()
    embedding = nomic_embed(model, tokenizer, query_text)

    logger.info("embed_query_done", dim=len(embedding))
    return {"query_embedding": embedding}


async def entity_search(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Search for entities using fulltext index in Neo4j.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with entities and source_qids.
    """
    configuration = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]
    query_text = last_message.content
    driver = neo4j_client.get_driver()

    results = await query_knowledge_graph(
        ENTITY_FULLTEXT_SEARCH,
        driver=driver,
        parameters={"query": query_text, "limit": configuration.retrieval_k},
    )

    # Deduplicate by entity ID
    seen_ids: set[str] = set()
    entities: list[dict] = []
    qids: list[str] = []
    for row in results:
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            entities.append(row)
            if row.get("qid"):
                qids.append(row["qid"])

    logger.info("entity_search_done", count=len(entities))
    return {"entities": entities, "source_qids": qids}


async def neighborhood_expand(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Expand entity neighborhood by traversing 1-hop relationships.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with relationships and source_qids.
    """
    configuration = Configuration.from_runnable_config(config)
    driver = neo4j_client.get_driver()
    limit_per_entity = configuration.retrieval_k

    all_relationships: list[dict] = []
    qids: list[str] = []

    for entity in state.entities:
        entity_id = entity.get("id")
        if not entity_id:
            continue
        results = await query_knowledge_graph(
            ENTITY_NEIGHBORHOOD,
            driver=driver,
            parameters={"entity_id": entity_id, "limit": limit_per_entity},
        )
        all_relationships.extend(results)
        for row in results:
            if row.get("target_qid"):
                qids.append(row["target_qid"])

    logger.info("neighborhood_expand_done", count=len(all_relationships))
    return {"relationships": all_relationships, "source_qids": qids}


async def chunk_search(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Search Pinecone chunks index for supporting text evidence.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with chunk_evidence.
    """
    configuration = Configuration.from_runnable_config(config)
    client = pinecone_client.get_client()

    # Build optional filter from entity QIDs
    entity_qids = [e.get("qid") for e in state.entities if e.get("qid")]
    filter_dict = {"article_qid": {"$in": entity_qids}} if entity_qids else None

    response = vector_search(
        query_vector=state.query_embedding,
        client=client,
        index_name=settings.pinecone_index_chunks_name,
        top_k=configuration.retrieval_k,
        filter_dict=filter_dict,
    )

    chunks = [
        {
            "id": match["id"],
            "score": match["score"],
            "text": match.get("metadata", {}).get("text", ""),
            "article_qid": match.get("metadata", {}).get("article_qid", ""),
        }
        for match in response.get("matches", [])
    ]

    logger.info("chunk_search_done", count=len(chunks))
    return {"chunk_evidence": chunks}


async def community_search(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Search Pinecone community-summaries index for community reports.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with community_reports.
    """
    configuration = Configuration.from_runnable_config(config)
    client = pinecone_client.get_client()

    filter_dict = {"level": {"$eq": configuration.community_level}}

    response = vector_search(
        query_vector=state.query_embedding,
        client=client,
        index_name=settings.pinecone_index_community_summaries,
        top_k=configuration.retrieval_k,
        filter_dict=filter_dict,
    )

    reports = [
        {
            "id": match["id"],
            "score": match["score"],
            "summary": match.get("metadata", {}).get("text", ""),
            "community_id": match.get("metadata", {}).get("community_id", ""),
            "level": match.get("metadata", {}).get("level", ""),
        }
        for match in response.get("matches", [])
    ]

    logger.info("community_search_done", count=len(reports))
    return {"community_reports": reports}


async def community_members_search(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Fetch entity members of discovered communities from Neo4j.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with entities and source_qids.
    """
    _ = config
    driver = neo4j_client.get_driver()

    community_ids = [
        r.get("community_id") for r in state.community_reports if r.get("community_id")
    ]

    all_entities: list[dict] = []
    qids: list[str] = []

    for community_id in community_ids:
        results = await query_knowledge_graph(
            COMMUNITY_MEMBERS,
            driver=driver,
            parameters={"community_id": community_id},
        )
        all_entities.extend(results)
        for row in results:
            if row.get("qid"):
                qids.append(row["qid"])

    logger.info("community_members_search_done", count=len(all_entities))
    return {"entities": all_entities, "source_qids": qids}


async def resolve_sources(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Resolve all collected QIDs to Wikipedia source URLs.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with source_urls.
    """
    _ = config
    parquet_path = f"{settings.data_volume_path}/wikipedia_articles.parquet"
    urls = resolve_source_urls(state.source_qids, parquet_path)

    logger.info("resolve_sources_done", count=len(urls))
    return {"source_urls": urls}
