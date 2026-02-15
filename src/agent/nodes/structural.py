"""Structural expert node for NL-to-Cypher translation.

Handles queries requiring precise database lookups, counts, or lists.
Uses EXPLAIN for validation and error-feedback loops.
"""

import asyncio
from pathlib import Path
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client, neo4j_client
from agent.settings import settings
from agent.state import State
from agent.tools.gemini import gemini_generate
from agent.tools.knowledge_graph import format_schema_for_prompt, query_knowledge_graph

logger = structlog.get_logger()

def _get_query_text(message: Any) -> str:
    """Extract string content from a message, handling list-of-blocks format."""
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
SCHEMA:
{schema}

Your task is to convert the user's Natural Language query into a precise Cypher query.

GUIDELINES:
1. Use ONLY the node labels and properties provided in the schema.
2. For entity lookups (Artists, Bands, Genres, Countries), ALWAYS prioritize using the FullText index:
   Example: CALL db.index.fulltext.queryNodes('entityNameIndex', 'techno') YIELD node AS g
   IMPORTANT: When using db.index.fulltext.queryNodes, ensure the search term is just the keywords (e.g., 'Kraftwerk' instead of 'where is Kraftwerk from?'). DO NOT include punctuation like '?' or trailing quotes inside the search string.
3. For lists of items, always use 'DISTINCT' and always return 'name', 'type', and 'qid' if available.
4. For counting queries, return a field named 'count'.
5. PRECISION FIRST: Favor direct relationships (e.g., -[:HAS_GENRE]->) initially. Only use variable-length paths (e.g., *1..2) in your retry attempt if the direct match returned 0 results.
6. When filtering properties without using the index, use toLower() and CONTAINS for resilience.
7. ALWAYS prefix your query with 'EXPLAIN' to allow for initial validation.
8. If an error message is provided below, analyze it and FIX your previous query.

{error_context}
User Question: {query}

Respond with ONLY the Cypher code block."""


async def nl_to_cypher(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Expert node for generating and executing Cypher queries with retries.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: State update with cypher results, entities, or errors.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    driver = await neo4j_client.get_driver()
    query_text = _get_query_text(state.messages[-1])
    schema_path = Path(settings.data_volume_path) / "graph_schema.json"
    schema = format_schema_for_prompt(schema_path)

    error_context = ""
    max_retries = 2

    for attempt in range(max_retries + 1):
        # 1. Generate Cypher
        prompt = STRUCTURAL_CYPHER_PROMPT.format(
            schema=schema, 
            query=query_text, 
            error_context=error_context
        )
        cypher_response = await asyncio.to_thread(
            gemini_generate, client, prompt, model=configuration.model
        )
        cypher_response = cypher_response.strip()
        
        # Clean markdown
        if "```cypher" in cypher_response:
            cypher_response = cypher_response.split("```cypher")[1].split("```")[0].strip()
        elif "```" in cypher_response:
            cypher_response = cypher_response.split("```")[1].split("```")[0].strip()

        # Ensure EXPLAIN prefix
        if not cypher_response.upper().startswith("EXPLAIN"):
            cypher_response = f"EXPLAIN {cypher_response}"

        logger.info("nl_to_cypher_attempt", attempt=attempt, cypher=cypher_response)

        # 2. Validate and Execute
        try:
            # First, run EXPLAIN to validate syntax and schema mapping
            await query_knowledge_graph(cypher_response, driver=driver)
            
            # If EXPLAIN passed, run the actual query
            actual_query = cypher_response.replace("EXPLAIN", "", 1).replace("explain", "", 1).strip()
            logger.info("nl_to_cypher_executing", query=actual_query)
            results = await query_knowledge_graph(actual_query, driver=driver)
            
            # 3. Handle 0-result case with internal retry
            if not results and attempt < max_retries:
                logger.warning("nl_to_cypher_zero_results", attempt=attempt)
                error_context = (
                    "PREVIOUS QUERY RETURNED 0 RESULTS.\n"
                    "Try using flexible relationship pathing (e.g., *1..2) or broader search "
                    "criteria to find the connections."
                )
                continue

            # 4. Process results into state entities and collect QIDs
            new_entities = []
            new_qids = []
            for row in results:
                # If the row has a 'name' field, treat the whole row as an entity
                if "name" in row:
                    new_entities.append(row)
                    if row.get("qid"):
                        new_qids.append(row["qid"])
                else:
                    # Fallback for complex results: check all values
                    for val in row.values():
                        if isinstance(val, dict):
                            if "name" in val or "id" in val:
                                new_entities.append(val)
                            if "qid" in val and val["qid"]:
                                new_qids.append(val["qid"])
                        elif not isinstance(val, (list, dict)):
                            new_entities.append({"name": str(val), "description": "Structural result"})
                
                # Also capture any top-level qid columns
                if "qid" in row and row["qid"] and row["qid"] not in new_qids:
                    new_qids.append(row["qid"])

            logger.info("nl_to_cypher_success", results_count=len(results), entities_count=len(new_entities))
            return {
                "generated_cypher": cypher_response,
                "cypher_result": results,
                "entities": new_entities,
                "source_qids": new_qids,
                "cypher_error": ""
            }

        except Exception as e:
            error_msg = str(e)
            logger.warning("structural_expert_failed", attempt=attempt, error=error_msg)
            error_context = f"PREVIOUS ERROR: {error_msg}\nPLEASE FIX THE QUERY."
            if attempt == max_retries:
                return {
                    "generated_cypher": cypher_response,
                    "cypher_error": f"Failed after {max_retries} attempts. Last error: {error_msg}"
                }

    return {"cypher_error": "Unexpected termination of retry loop"}
