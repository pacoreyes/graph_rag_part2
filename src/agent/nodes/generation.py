"""Generation nodes for the LangGraph agent.

LLM-powered nodes for query analysis and answer synthesis.
"""

from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client
from agent.settings import settings
from agent.state import State
from agent.tools.gemini import gemini_generate
from agent.tools.knowledge_graph import format_schema_for_prompt

logger = structlog.get_logger()

PLANNER_PROMPT = (
    "You are the Lead Planner for a music-domain GraphRAG agent.\n"
    "{schema}\n\n"
    "Your goal is to analyze the user query and create a multi-step retrieval and reasoning plan.\n\n"
    "STRATEGIES:\n"
    '- "local": Use when the query is about specific entities and their direct properties.\n'
    '- "global": Use for broad themes, movements, or trends spanning many entities.\n'
    '- "drift": Use for multi-hop reasoning or following chains of influence.\n'
    '- "structural": Use for precise lists, counts, or specific database filters.\n'
    '- "hybrid": Use when multiple strategies are needed for full coverage.\n\n'
    "JSON RESPONSE FORMAT:\n"
    "{{\n"
    '  "strategy": "...",\n'
    '  "plan": "Detailed description of the steps you will take and why.",\n'
    '  "target_entity_types": [...],\n'
    '  "expected_outcome": "Description of what a successful answer should contain (e.g., \'A list of exactly 10 bands\')."\n'
    "}}\n\n"
    "User query: {query}\n\n"
    "Respond with ONLY the JSON object."
)

STRATEGY_PROMPT = (
    "You are a query analyzer for a music-domain GraphRAG system.\n"
    "{schema}\n\n"
    "Analyze the following user query and provide a JSON response with 'strategy' and 'target_entity_types'.\n\n"
    "STRATEGIES:\n"
    '- "local": The query is about specific entities (artists, albums, cities) '
    "and their direct properties.\n"
    '- "global": The query asks about broad themes, movements, genres, or trends '
    "spanning many entities.\n"
    '- "drift": The query requires multi-hop reasoning or following chains of '
    "relationships.\n"
    '- "hybrid": The query combines specific entity questions with broader thematic context.\n'
    '- "structural": The query asks for lists, counts, or specific properties '
    "that can be answered by a direct database query (e.g., 'List 10 techno bands from UK').\n\n"
    "TARGET ENTITY TYPES:\n"
    "Identify which entity types from the schema are most relevant to the query to filter the search. "
    "If the query is broad, return an empty list.\n\n"
    "User query: {query}\n\n"
    "Respond with ONLY a JSON object: {{\"strategy\": \"...\", \"target_entity_types\": [...]}}"
)

SYNTHESIS_PROMPT = (
    "You are a knowledgeable music assistant answering questions using a "
    "combination of community thematic summaries, knowledge graph facts, "
    "original source text, and the underlying graph schema.\n\n"
    "{schema}\n\n"
    "THEMATIC CONTEXT (from community reports):\n"
    "{community_reports}\n\n"
    "SPECIFIC FACTS (from knowledge graph):\n"
    "{entities}\n"
    "{relationships}\n\n"
    "SUPPORTING EVIDENCE (from source text):\n"
    "{chunks}\n\n"
    "STRUCTURAL RESULTS (from database query):\n"
    "{cypher_result}\n"
    "{cypher_error}\n\n"
    "CRITIQUE FROM PREVIOUS ATTEMPT (if any):\n"
    "{critique}\n\n"
    "RETRIEVAL GUIDE (Instructions for focus):\n"
    "{guide}\n\n"
    "SOURCES (Wikipedia Metadata):\n"
    "{sources}\n\n"
    "Using the above, answer the user's question: {query}\n\n"
    "Guidelines:\n"
    "1. CROSS-VERIFICATION (TRIANGULATION): Use 'STRUCTURAL RESULTS' from the database as your primary candidates, but you MUST verify them against 'SUPPORTING EVIDENCE' (text chunks) and 'SPECIFIC FACTS' (entity descriptions).\n"
    "2. TRUTH HIERARCHY: If a database result is contradicted by a text chunk (e.g., the database says a band is German, but Wikipedia text says they are Swedish), EXCLUDE the band and explain the discrepancy.\n"
    "3. COMPLETENESS: Aim to include all verified items found in the 'STRUCTURAL RESULTS'. Do not include unverified or contradicted items.\n"
    "4. SUPPLEMENTARY EVIDENCE: Use 'THEMATIC CONTEXT' only to add broad framing to the verified facts.\n"
    "5. Provide a comprehensive, well-structured answer in plain text.\n"
    "6. DO NOT include Wikipedia links or bracketed citations [Source Name] inside the main text of your answer.\n"
    "7. After your answer, add a horizontal line (---) followed by a 'Sources:' header and list the specific evidence used to form the response.\n"
    "8. Categorize sources as follows:\n"
    "   - **Graph Facts**: List key Entities and Relationships/Paths used.\n"
    "   - **Thematic Context**: Mention which Community IDs and Titles provided broad framing.\n"
    "   - **Text Evidence**: List Wikipedia Names and their URLs for the supporting chunks used.\n"
    "   - **Database Query**: If a structural query was used, mention it and explain if it succeeded or failed.\n"
    "9. For 'drift' queries, specifically describe the chains of connections found in the Graph Facts section.\n"
    "10. If the context does not contain enough information, say so clearly."
)

CYPHER_GENERATION_PROMPT = (
    "You are a Cypher query generator for a music GraphRAG system.\n"
    "SCHEMA:\n{schema}\n\n"
    "Write a Cypher query to answer the following user question.\n"
    "Guidelines:\n"
    "1. Use the provided node types and properties from the schema.\n"
    "2. Most text search should use 'CONTAINS' or 'db.index.fulltext.queryNodes' on 'entityNameIndex'.\n"
    "3. ALWAYS start the query with 'EXPLAIN' so the system can validate it without execution first.\n"
    "4. Return node properties like 'name', 'description', 'type', and 'qid'.\n"
    "5. User query: {query}\n\n"
    "Respond with ONLY the Cypher query block."
)


def _format_entities(entities: list[dict]) -> str:
    """Format entity records into a readable context block.

    Args:
        entities: List of entity dicts with name, description fields.

    Returns:
        str: Formatted entities section.
    """
    if not entities:
        return ""
    lines = ["## Entities"]
    for e in entities:
        name = e.get("name", "Unknown")
        desc = e.get("description", "No description")
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)


def _format_relationships(relationships: list[dict]) -> str:
    """Format relationship records into a readable context block.

    Args:
        relationships: List of relationship dicts.

    Returns:
        str: Formatted relationships section.
    """
    if not relationships:
        return ""
    lines = ["## Relationships"]
    for r in relationships:
        source = r.get("source_name", "?")
        rel = r.get("relationship", "RELATED_TO")
        target = r.get("target_name", "?")
        desc = r.get("rel_description", "")
        line = f"- {source} --[{rel}]--> {target}"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def _format_community_reports(reports: list[dict]) -> str:
    """Format community report records into a readable context block.

    Args:
        reports: List of community report dicts.

    Returns:
        str: Formatted community reports section.
    """
    if not reports:
        return ""
    lines = ["## Community Reports"]
    for r in reports:
        cid = r.get("community_id", "?")
        title = r.get("title", "")
        summary = r.get("summary", "No summary")
        header = f"Community {cid}"
        if title:
            header += f": {title}"
        lines.append(f"- {header}: {summary}")
    return "\n".join(lines)


def _format_chunks(chunks: list[dict]) -> str:
    """Format text chunk records into a readable context block.

    Args:
        chunks: List of chunk dicts with text and score.

    Returns:
        str: Formatted chunks section.
    """
    if not chunks:
        return ""
    lines = ["## Text Evidence"]
    for c in chunks:
        score = c.get("score", 0)
        text = c.get("text", "")
        article_id = c.get("article_id", "")
        line = f"- [score={score:.3f}, source={article_id}] {text}"
        lines.append(line)
    return "\n".join(lines)


def _format_sources(source_urls: dict[str, dict[str, str]]) -> str:
    """Format source URL records into a readable context block.

    Args:
        source_urls: Mapping of QID to name/URL dicts.

    Returns:
        str: Formatted sources section.
    """
    if not source_urls:
        return ""
    lines = ["## Sources"]
    for qid, info in source_urls.items():
        name = info.get("name", qid)
        url = info.get("wikipedia_url", "")
        if url:
            lines.append(f"- [{name}]({url})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


async def planner(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Analyze query and generate a structured retrieval and reasoning plan.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: State update with strategy, plan, target_entity_types, and expected_outcome.
    """
    configuration = Configuration.from_runnable_config(config)
    client = gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = last_message.content

    schema = format_schema_for_prompt(Path(settings.data_volume_path) / "graph_schema.json")
    prompt = PLANNER_PROMPT.format(schema=schema, query=query_text)

    response_text = gemini_generate(client, prompt, model=configuration.model).strip()
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        import json
        data = json.loads(response_text)
        strategy = data.get("strategy", "hybrid").lower()
        plan = data.get("plan", "")
        target_entity_types = data.get("target_entity_types", [])
        expected_outcome = data.get("expected_outcome", "")
    except Exception:
        logger.warning("planner_parse_failed", raw=response_text)
        strategy = "hybrid"
        plan = "Fallback to hybrid search."
        target_entity_types = []
        expected_outcome = ""

    if strategy not in ("local", "global", "drift", "hybrid", "structural"):
        strategy = "hybrid"

    logger.info("planner_done", strategy=strategy, plan=plan)
    return {
        "strategy": strategy, 
        "plan": plan, 
        "target_entity_types": target_entity_types,
        "iteration_count": 0 # Reset loop counter on new question
    }


async def query_analyzer(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user query into a search strategy and identify target types.

    Calls Gemini to determine search strategy and relevant entity types.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with strategy and target_entity_types.
    """
    configuration = Configuration.from_runnable_config(config)
    client = gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = last_message.content

    schema_path = Path(settings.data_volume_path) / "graph_schema.json"
    schema = format_schema_for_prompt(schema_path)
    prompt = STRATEGY_PROMPT.format(schema=schema, query=query_text)

    # Use JSON response if supported or strip markdown
    response_text = gemini_generate(client, prompt, model=configuration.model).strip()
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        import json

        data = json.loads(response_text)
        strategy = data.get("strategy", "hybrid").lower()
        target_entity_types = data.get("target_entity_types", [])
    except Exception:
        logger.warning("query_analyzer_parse_failed", raw=response_text)
        strategy = "hybrid"
        target_entity_types = []

    # Validate to one of the four strategies
    if strategy not in ("local", "global", "drift", "hybrid"):
        strategy = "hybrid"

    logger.info("query_analyzer_done", strategy=strategy, types=target_entity_types)
    return {"strategy": strategy, "target_entity_types": target_entity_types}


async def cypher_generator(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Generate a Cypher query based on the user question and schema.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with generated_cypher.
    """
    configuration = Configuration.from_runnable_config(config)
    client = gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = last_message.content

    schema = format_schema_for_prompt(Path(settings.data_volume_path) / "graph_schema.json")
    prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, query=query_text)

    response_text = gemini_generate(client, prompt, model=configuration.model).strip()
    
    # Clean up markdown code blocks
    if "```cypher" in response_text:
        response_text = response_text.split("```cypher")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    logger.info("cypher_generator_done", query=response_text)
    return {"generated_cypher": response_text}


async def synthesize_answer(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Synthesize a final answer from all retrieved context.

    Builds a structured prompt from entities, relationships, community
    reports, chunks, and sources, then calls Gemini to generate a cited
    answer.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with AIMessage answer.
    """
    configuration = Configuration.from_runnable_config(config)
    client = gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = last_message.content

    # Build structured prompt with separate context sections
    schema_path = Path(settings.data_volume_path) / "graph_schema.json"
    schema = format_schema_for_prompt(schema_path)
    prompt = SYNTHESIS_PROMPT.format(
        schema=schema,
        community_reports=_format_community_reports(state.community_reports[:3]),
        entities=_format_entities(state.entities[:10]),
        relationships=_format_relationships(state.relationships[:15]),
        chunks=_format_chunks(state.chunk_evidence[:5]),
        cypher_result=str(state.cypher_result[:20]),
        cypher_error=state.cypher_error,
        critique=state.critique,
        guide=state.retrieval_guide,
        sources=_format_sources(state.source_urls),
        query=query_text,
    )
    answer = gemini_generate(client, prompt, model=configuration.model)

    logger.info("synthesize_answer_done", length=len(answer))
    return {"messages": [AIMessage(content=answer)]}
