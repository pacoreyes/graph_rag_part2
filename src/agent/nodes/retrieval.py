# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Retrieval nodes for the LangGraph agent.
# Each function is a LangGraph node that performs one step in
# the multi-strategy retrieval pipeline (Local / Global / Hybrid).
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import asyncio
import math
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import (
    neo4j_client,
    hf_embedding_client,
    pinecone_client,
)
from agent.settings import settings
from agent.state import State
from agent.tools.embeddings import process_embedding
from agent.tools.knowledge_graph import (
    query_knowledge_graph,
    sanitize_lucene_query,
)
from agent.tools.source_resolver import resolve_source_urls
from agent.tools.vector_store import pinecone_embed, vector_search

logger = structlog.get_logger()

# --- Cypher queries ---

ENTITY_FULLTEXT_SEARCH = (
    "CALL db.index.fulltext.queryNodes('entityNameIndex', $query) "
    "YIELD node, score "
    "RETURN node.id AS id, node.name AS name, node.description AS description, "
    "node.type AS type, node.qid AS qid, node.pagerank AS pagerank, score "
    "ORDER BY score DESC LIMIT $limit"
)

ENTITY_VECTOR_SEARCH = (
    "CALL db.index.vector.queryNodes('entity_vector_index', $limit, $embedding) "
    "YIELD node, score "
    "RETURN node.id AS id, node.name AS name, node.description AS description, "
    "node.type AS type, node.qid AS qid, node.pagerank AS pagerank, score"
)

# Semantic Relation Search (DRIFT) - find semantically relevant neighbors
SEMANTIC_RELATION_SEARCH = (
    "MATCH (e {id: $entity_id})-[r]->(neighbor) "
    "WITH e, neighbor, r, vector.similarity.cosine(r.embedding, $embedding) AS score "
    "WHERE score > $threshold "
    "RETURN e.name AS source_name, type(r) AS relationship, r.description AS rel_description, "
    "neighbor.id AS id, neighbor.name AS name, neighbor.description AS description, "
    "neighbor.type AS type, neighbor.qid AS qid, score "
    "ORDER BY score DESC LIMIT $limit"
)

# Property-based community membership (replaces IN_COMMUNITY edges)
COMMUNITY_MEMBERS_L0 = (
    "MATCH (e) WHERE e.community_L0 = $community_id "
    "RETURN e.id AS id, e.name AS name, e.description AS description, "
    "e.type AS type, e.qid AS qid, e.pagerank AS pagerank "
    "ORDER BY e.pagerank DESC LIMIT 20"
)

COMMUNITY_MEMBERS_L1 = (
    "MATCH (e) WHERE e.community_L1 = $community_id "
    "RETURN e.id AS id, e.name AS name, e.description AS description, "
    "e.type AS type, e.qid AS qid, e.pagerank AS pagerank "
    "ORDER BY e.pagerank DESC LIMIT 20"
)

COMMUNITY_MEMBERS_L2 = (
    "MATCH (e) WHERE e.community_L2 = $community_id "
    "RETURN e.id AS id, e.name AS name, e.description AS description, "
    "e.type AS type, e.qid AS qid, e.pagerank AS pagerank "
    "ORDER BY e.pagerank DESC LIMIT 20"
)


def _get_query_text(message: Any) -> str:
    """Extract string content from a message, handling list-of-blocks format."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from the first text block found
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
            if isinstance(block, str):
                return block
    return str(content)


async def embed_query(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Compute Snowflake embedding (384-dim) for the user query using HF Inference API.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with query_embedding.
    """
    _ = config
    last_message = state.messages[-1]
    query_text = _get_query_text(last_message)

    # Call HF Inference API via HFEmbeddingClient
    raw_embedding = await hf_embedding_client.embed(query_text)
    
    # process_embedding handles L2 normalization
    embedding = process_embedding(raw_embedding)

    logger.info("embed_query_done", dim=len(embedding))
    return {"query_embedding": embedding}


async def entity_search(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Search for entities using both fulltext and vector indexes in Neo4j.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with deduplicated entities and source_qids.
    """
    configuration = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]
    query_text = _get_query_text(last_message)
    driver = await neo4j_client.get_driver()

    # 1. Fulltext search (sanitized for Lucene)
    sanitized_query = sanitize_lucene_query(query_text)
    ft_results = await query_knowledge_graph(
        ENTITY_FULLTEXT_SEARCH,
        driver=driver,
        parameters={"query": sanitized_query, "limit": configuration.retrieval_k},
    )

    # 2. Vector search (requires query_embedding from embed_query node)
    v_results = await query_knowledge_graph(
        ENTITY_VECTOR_SEARCH,
        driver=driver,
        parameters={
            "embedding": state.query_embedding,
            "limit": configuration.retrieval_k,
        },
    )

    # 3. Deduplicate and merge by entity ID, applying similarity filter for vector results
    seen_ids: set[str] = set()
    seen_names: dict[str, float] = {}  # name -> pagerank
    entities: list[dict] = []
    qids: list[str] = []

    # Combine results
    all_results = ft_results + [
        r for r in v_results if r.get("score", 0) > configuration.similarity_threshold
    ]

    # Sort combined results by PageRank (desc) then Score (desc) to prioritize canonical entities
    all_results.sort(key=lambda x: (x.get("pagerank", 0), x.get("score", 0)), reverse=True)

    # Compute top raw score BEFORE deduplication for correct pruning
    top_score = max(
        (r.get("score", 0) for r in all_results), default=0
    )

    for row in all_results:
        eid = row["id"]
        name = row["name"]
        pagerank = row.get("pagerank", 0)

        # Skip if we've seen this ID
        if eid in seen_ids:
            continue

        # Name-based deduplication: skip if we've seen this name with a higher PageRank
        if name in seen_names and seen_names[name] >= pagerank:
            continue

        # Relative score pruning: skip entities below 80% of top raw score
        if top_score > 0 and row.get("score", 0) < top_score * 0.8:
            continue

        seen_ids.add(eid)
        seen_names[name] = pagerank
        # Add origin metadata for USA Framework
        row["origin"] = "Graph DB"
        row["method"] = "Entity Search (Fulltext/Vector)"
        # row["type"] is preserved from Neo4j result
        entities.append(row)
        if row.get("qid"):
            qids.append(row["qid"])

    # Soft filter by target_entity_types: prefer matching types but keep
    # non-matching entities if filtered set would be too small
    target_types = state.target_entity_types
    if target_types:
        typed = [e for e in entities if e.get("type") in target_types]
        if len(typed) >= configuration.retrieval_k:
            entities = typed

    # Limit to top K total entities to keep context focused
    entities = entities[:configuration.retrieval_k * 2]

    logger.info("entity_search_done", ft_count=len(ft_results), v_count=len(v_results), total=len(entities))
    return {"entities": entities, "source_qids": qids}


async def neighborhood_expand(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Expand entity neighborhood using semantic relation traversal (DRIFT).

    Traverses multiple hops (based on configuration) along relationships
    that are semantically relevant to the user query.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with relationships and source_qids.
    """
    configuration = Configuration.from_runnable_config(config)
    driver = await neo4j_client.get_driver()
    limit_per_entity = configuration.retrieval_k
    max_depth = configuration.neighborhood_depth

    all_relationships: list[dict] = []
    all_qids: list[str] = []
    visited_ids: set[str] = {e.get("id") for e in state.entities if e.get("id")}
    current_entities = list(state.entities)

    for depth in range(max_depth):
        # Query all entities at this depth concurrently
        entity_ids = [e.get("id") for e in current_entities if e.get("id")]
        if not entity_ids:
            break

        hop_results = await asyncio.gather(*(
            query_knowledge_graph(
                SEMANTIC_RELATION_SEARCH,
                driver=driver,
                parameters={
                    "entity_id": eid,
                    "embedding": state.query_embedding,
                    "threshold": configuration.similarity_threshold,
                    "limit": limit_per_entity,
                },
            )
            for eid in entity_ids
        ))

        next_entities = []
        for results in hop_results:
            max_rel_score = results[0].get("score", 0) if results else 0
            for row in results:
                score = row.get("score", 0)

                # Relative score pruning for neighbors within a hop
                if score < max_rel_score * 0.8:
                    continue

                row["origin"] = "Graph DB"
                row["method"] = "Neighborhood Expansion (DRIFT)"

                all_relationships.append(row)
                if row.get("qid"):
                    all_qids.append(row["qid"])

                neighbor_id = row.get("id")
                if neighbor_id and neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    next_entities.append(row)

        if not next_entities:
            break
        current_entities = next_entities

    logger.info(
        "neighborhood_expand_done",
        depth=max_depth,
        rel_count=len(all_relationships),
        new_entities=len(visited_ids) - len(state.entities),
    )
    return {"relationships": all_relationships, "source_qids": all_qids}


async def chunk_search(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Search Pinecone chunks index for supporting text evidence using Surgical Pattern.

    Implements:
    1. Semantic Scoping (Filter by entity_type)
    2. Implicit Evidence Discovery (Expand by community_id)
    3. Authority-Led Ranking (Re-rank by pagerank)

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with chunk_evidence.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await pinecone_client.get_client()
    query_text = state.messages[-1].content

    # Generate Llama embedding explicitly (1024-dim)
    query_vector = await asyncio.to_thread(pinecone_embed, client, query_text)

    # 1. Build Multi-Factor Filter
    filters = []
    
    # Discovery by Entity ID (Discovered entities)
    entity_qids = [e.get("qid") for e in state.entities if e.get("qid")]
    if entity_qids:
        filters.append({"article_id": {"$in": entity_qids}})

    # Discovery by Community (Scene-level expansion)
    community_ids = [
        r.get("community_id") for r in state.community_reports 
        if r.get("community_id") is not None
    ]
    if community_ids:
        filters.append({"community_id": {"$in": community_ids}})

    # Combine ID/Community filters with OR
    if len(filters) > 1:
        filter_dict = {"$or": filters}
    elif filters:
        filter_dict = filters[0]
    else:
        filter_dict = None

    # 2. Build graduated filter cascade (strict → loose)
    type_filter = (
        {"entity_type": {"$in": state.target_entity_types}}
        if state.target_entity_types
        else None
    )

    filter_cascade: list[dict | None] = []
    # Level 1: entity_type AND (article_id OR community_id)
    if type_filter and filter_dict:
        filter_cascade.append({"$and": [type_filter, filter_dict]})
    # Level 2: article_id OR community_id (without type constraint)
    if filter_dict:
        filter_cascade.append(filter_dict)
    # Level 3: entity_type only
    if type_filter:
        filter_cascade.append(type_filter)
    # Level 4: open search (no filter)
    filter_cascade.append(None)

    k = configuration.retrieval_k
    response: dict = {"matches": []}
    for level, cascade_filter in enumerate(filter_cascade):
        response = await asyncio.to_thread(
            vector_search,
            client=client,
            index_name=settings.pinecone_index_chunks_name,
            query_vector=query_vector,
            top_k=k * 2 if cascade_filter else k,
            filter_dict=cascade_filter,
        )
        if response.get("matches"):
            break
        logger.warning(
            "chunk_search_fallback",
            reason="no_filtered_results",
            level=level,
            filter=cascade_filter,
        )

    # 3. Authority-Led Re-ranking (Anchor Retrieval)
    matches = response.get("matches", [])
    chunks = []
    qids = []
    for match in matches:
        metadata = match.get("metadata", {})
        score = match["score"]
        
        # Apply pagerank boost: score * log1p(pagerank)
        # Assuming pagerank is a float in metadata
        pagerank = float(metadata.get("pagerank", 0))
        boosted_score = score * math.log1p(pagerank)
        
        article_id = metadata.get("article_id", "")
        if article_id:
            qids.append(article_id)

        chunks.append({
            "id": match["id"],
            "score": boosted_score,
            "raw_score": score,
            "text": metadata.get("text", ""),
            "article_id": article_id,
            "entity_type": metadata.get("entity_type", ""),
            "community_id": metadata.get("community_id", ""),
            "pagerank": pagerank,
            "origin": "Vector DB",
            "method": "Surgical Search",
            "type": "Text Chunk"
        })

    # Sort by boosted score and limit to retrieval_k
    chunks.sort(key=lambda x: x["score"], reverse=True)
    chunks = chunks[:k]

    logger.info("chunk_search_done", count=len(chunks), filter=bool(filter_dict))
    return {"chunk_evidence": chunks, "source_qids": qids}


async def community_search(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Search Pinecone community-summaries index for community reports using Integrated Inference.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with community_reports.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await pinecone_client.get_client()
    query_text = state.messages[-1].content

    # Generate Llama embedding explicitly (1024-dim)
    query_vector = await asyncio.to_thread(pinecone_embed, client, query_text)

    # Build level filter — single call with $or when multi-level
    levels_to_search = [configuration.community_level]
    if configuration.community_level == 2 and state.strategy in ("global", "hybrid"):
        levels_to_search.append(1)

    if len(levels_to_search) > 1:
        filter_dict = {"$or": [{"level": {"$eq": lv}} for lv in levels_to_search]}
    else:
        filter_dict = {"level": {"$eq": levels_to_search[0]}}

    response = await asyncio.to_thread(
        vector_search,
        client=client,
        index_name=settings.pinecone_index_community_summaries,
        query_vector=query_vector,
        top_k=configuration.retrieval_k * len(levels_to_search),
        filter_dict=filter_dict,
    )

    all_reports = [
        {
            "id": match["id"],
            "score": match["score"],
            "summary": match.get("metadata", {}).get("text", ""),
            "title": match.get("metadata", {}).get("title", ""),
            "community_id": match.get("metadata", {}).get("community_id", ""),
            "level": match.get("metadata", {}).get("level", ""),
            "origin": "Vector DB",
            "method": f"Community Summary (Leiden Level {match.get('metadata', {}).get('level', '')})",
            "type": "Thematic Summary",
        }
        for match in response.get("matches", [])
    ]

    logger.info("community_search_done", count=len(all_reports), levels=levels_to_search)
    return {"community_reports": all_reports}


async def community_members_search(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Fetch entity members of discovered communities from Neo4j using property-based lookup.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state with entities and source_qids.
    """
    _ = config
    driver = await neo4j_client.get_driver()

    # Build query tasks for all communities
    _level_queries = {0: COMMUNITY_MEMBERS_L0, 1: COMMUNITY_MEMBERS_L1}
    tasks = []
    task_levels: list[int] = []
    for report in state.community_reports:
        community_id = report.get("community_id")
        level = report.get("level")
        if community_id is None or level is None:
            continue
        query = _level_queries.get(level, COMMUNITY_MEMBERS_L2)
        tasks.append(
            query_knowledge_graph(
                query, driver=driver, parameters={"community_id": community_id},
            )
        )
        task_levels.append(level)

    # Execute all community queries concurrently
    all_results = await asyncio.gather(*tasks) if tasks else []

    all_entities: list[dict] = []
    qids: list[str] = []
    for level, results in zip(task_levels, all_results):
        for row in results:
            row["origin"] = "Graph DB"
            row["method"] = f"Community Discovery (L{level})"
        all_entities.extend(results)
        for row in results:
            if row.get("qid"):
                qids.append(row["qid"])

    # Soft filter by target_entity_types: prefer matching types
    target_types = state.target_entity_types
    if target_types:
        typed = [e for e in all_entities if e.get("type") in target_types]
        if typed:
            all_entities = typed
            qids = [e["qid"] for e in all_entities if e.get("qid")]

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
    urls = await asyncio.to_thread(resolve_source_urls, state.source_qids, parquet_path)

    logger.info("resolve_sources_done", count=len(urls))
    return {"source_urls": urls}
