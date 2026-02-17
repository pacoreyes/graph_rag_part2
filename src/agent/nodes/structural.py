# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Structural expert node for NL-to-Cypher translation.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Structural expert node for NL-to-Cypher translation.

Handles queries requiring precise database lookups, counts, or lists.
Uses EXPLAIN for validation and error-feedback loops.
"""

import asyncio
import json
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client, neo4j_client
from agent.nodes.models import CypherResponse
from agent.state import State
from agent.tools.gemini import gemini_generate
from agent.tools.knowledge_graph import query_knowledge_graph

logger = structlog.get_logger()


def _get_query_text(message: Any) -> str:
    """Extract string content from a message, handling list-of-blocks format.
    
    Args:
        message: LangChain message object.
        
    Returns:
        str: Extracted text content.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
            if isinstance(block, str):
                return block
    return str(content)


STRUCTURAL_CYPHER_PROMPT = """You are an expert Neo4j Cypher developer for a music GraphRAG system.

Your task is to convert the user's Natural Language query into a precise Cypher query.

GUIDELINES:
1. Use ONLY the node labels and properties provided in the schema.
2. For entity lookups (Artists, Bands, Genres, Countries), ALWAYS prioritize using the FullText index:
   Example: CALL db.index.fulltext.queryNodes('entityNameIndex', 'techno') YIELD node AS g
3. For lists of items, always use 'DISTINCT' and always return 'name', 'type', and 'qid' if available.
4. For counting queries, return a field named 'count'.
5. When filtering properties without using the index, use toLower() and CONTAINS for resilience.

{error_context}
User Question: {query}"""


async def nl_to_cypher(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Expert node for generating and executing Cypher queries with structured output and caching.
    
    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.
        
    Returns:
        dict[str, Any]: Partial state update with generated cypher and results.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    driver = await neo4j_client.get_driver()
    query_text = _get_query_text(state.messages[-1])
    
    system_instruction = gemini_client.get_schema_instruction()
    error_context, max_retries = "", 1

    clean_query = ""
    for attempt in range(max_retries + 1):
        prompt = STRUCTURAL_CYPHER_PROMPT.format(query=query_text, error_context=error_context)
        response = await asyncio.to_thread(
            gemini_generate, 
            client, 
            prompt, 
            model=configuration.model,
            response_schema=CypherResponse,
            system_instruction=system_instruction,
            response_mime_type="application/json",
        )
        
        if isinstance(response, CypherResponse):
            clean_query = response.cypher.strip()
        elif isinstance(response, dict):
            clean_query = response.get("cypher", "").strip()
        else:
            logger.warning("nl_to_cypher_parse_failed", raw=response)
            clean_query = str(response).strip() # Fallback

        logger.info("nl_to_cypher_validate", attempt=attempt, query=clean_query)

        try:
            # 2. Guardrail: Run EXPLAIN first
            await query_knowledge_graph(f"EXPLAIN {clean_query}", driver=driver)
            
            # 3. If EXPLAIN passed, run actual
            results = await query_knowledge_graph(clean_query, driver=driver)
            
            new_entities, new_qids = [], []
            for row in results:
                if "name" in row:
                    new_entities.append(row)
                    if row.get("qid"): new_qids.append(row["qid"])
                else:
                    for val in row.values():
                        if isinstance(val, dict):
                            if "name" in val or "id" in val: new_entities.append(val)
                            if "qid" in val and val["qid"]: new_qids.append(val["qid"])
                        elif not isinstance(val, (list, dict)):
                            new_entities.append({"name": str(val), "description": "Structural result"})
                if "qid" in row and row["qid"] and row["qid"] not in new_qids: new_qids.append(row["qid"])

            logger.info("nl_to_cypher_success", results_count=len(results))
            return {"generated_cypher": clean_query, "cypher_result": results, "entities": new_entities, "source_qids": new_qids, "cypher_error": ""}

        except Exception as e:
            error_msg = str(e)
            logger.warning("nl_to_cypher_validation_failed", attempt=attempt, error=error_msg)
            error_context = f"PREVIOUS ERROR: {error_msg}\nPLEASE FIX THE QUERY."
            if attempt == max_retries:
                return {"generated_cypher": clean_query, "cypher_error": error_msg, "cypher_result": []}

    return {"cypher_error": "Failed after retries"}
