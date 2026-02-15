"""Generation nodes for the LangGraph agent.

LLM-powered nodes for query analysis and answer synthesis.
"""

import asyncio
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
    "FAST TRACK:\n"
    "Set 'is_fast_track' to true ONLY if the query is a simple factoid question about a single entity "
    "that can likely be answered by a direct lookup (e.g., 'Where is Kraftwerk from?', 'Who is the "
    "lead singer of The Cure?'). Simple 'local' or 'structural' queries are good candidates.\n\n"
    "JSON RESPONSE FORMAT:\n"
    "{{\n"
    '  "strategy": "...",\n'
    '  "is_fast_track": true/false,\n'
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
    "UNIFIED CONTEXT (Atomic Knowledge Units):\n"
    "{akus}\n\n"
    "CRITIQUE FROM PREVIOUS ATTEMPT (if any):\n"
    "{critique}\n\n"
    "RETRIEVAL GUIDE (Instructions for focus):\n"
    "{guide}\n\n"
    "Using the above, answer the user's question: {query}\n\n"
    "Guidelines:\n"
    "1. ATTENTION PINNING: Every claim MUST be immediately followed by the relevant index(es) in brackets (e.g., [1] or [1][3]).\n"
    "2. TRUTH HIERARCHY: Prioritize AKUs from 'Graph DB' for structural facts and 'Vector DB' for descriptive context.\n"
    "3. VERBATIM EVIDENCE: For every 'Vector DB' AKU you cite, you MUST provide a verbatim quote from its text to prove the claim.\n"
    "4. NO HALLUCINATION: ONLY use numerical indices from the provided context list.\n"
    "5. CLEAN SYNTHESIS: Provide the answer in plain text without markdown links, URLs, or markdown headers (e.g., # or ##).\n"
    "6. JSON RESPONSE FORMAT:\n"
    "{{\n"
    '  "answer": "Your synthesis with [N] citations...",\n'
    '  "evidence": {{\n'
    '     "1": "Verbatim excerpt for AKU 1",\n'
    '     "2": "Relationship or fact summary for AKU 2",\n'
    '     ...\n'
    '  }}\n'
    "}}\n\n"
    "Respond with ONLY the JSON object."
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


def _flatten_dict(d: dict) -> str:
    """Flatten a dictionary into a comma-separated string of key-values.

    Used for making Cypher results human-readable in the context.
    """
    parts = []
    for k, v in d.items():
        if v is not None:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)


def homogenize_context(state: State) -> list[dict]:
    """Unify all retrieval results into a single list of Atomic Knowledge Units (AKUs).

    Applies deterministic sorting and indexing for the USA Framework with deduplication.

    Args:
        state: The current graph state.

    Returns:
        list[dict]: A list of indexed AKUs.
    """
    raw_akus = []

    # 1. Process Entities (Sort by PageRank)
    entities = sorted(
        state.entities, key=lambda x: x.get("pagerank") or 0, reverse=True
    )
    for e in entities:
        name = e.get("name", "Unknown")
        desc = e.get("description", "No description")
        raw_akus.append({
            "content": f"Entity: {name} - {desc}",
            "origin": e.get("origin", "Graph DB"),
            "method": e.get("method", "Entity Search"),
            "metadata": {"qid": e.get("qid"), "type": "Node", "name": name}
        })

    # 2. Process Relationships (Sort by Score)
    relationships = sorted(
        state.relationships, key=lambda x: x.get("score") or 0, reverse=True
    )
    for r in relationships:
        source = r.get("source_name", "?")
        rel = r.get("relationship", "RELATED_TO")
        # Fix: The neighbor search returns 'name' for the target entity
        target = r.get("target_name") or r.get("name", "?")
        desc = r.get("rel_description", "")
        content = f"Relationship: {source} --[{rel}]--> {target}"
        if desc:
            content += f" ({desc})"
        raw_akus.append({
            "content": content,
            "origin": r.get("origin", "Graph DB"),
            "method": r.get("method", "Neighborhood Expansion"),
            "metadata": {"qid": r.get("qid"), "type": "Relationship", "name": f"{source} --[{rel}]--> {target}"}
        })

    # 3. Process Text Chunks (Sort by Score)
    chunks = sorted(
        state.chunk_evidence, key=lambda x: x.get("score") or 0, reverse=True
    )
    for c in chunks:
        content = c.get("text", "")
        raw_akus.append({
            "content": f"Text Evidence: {content}",
            "origin": c.get("origin", "Vector DB"),
            "method": c.get("method", "Surgical Search"),
            "metadata": {
                "article_id": c.get("article_id"), 
                "chunk_id": c.get("id"), # Include the specific Pinecone ID
                "type": "Text Chunk"
            }
        })

    # 4. Process Community Reports (Sort by Score)
    reports = sorted(
        state.community_reports, key=lambda x: (x.get("level") or 0, x.get("score") or 0), reverse=True
    )
    for r in reports:
        cid = r.get("community_id", "?")
        title = r.get("title", "")
        summary = r.get("summary", "No summary")
        raw_akus.append({
            "content": f"Thematic Summary (Community {cid}): {title} - {summary}",
            "origin": r.get("origin", "Vector DB"),
            "method": r.get("method", "Community Search"),
            "metadata": {"community_id": cid, "type": "Community Report", "name": f"Community {cid}: {title}"}
        })

    # 5. Process Cypher Results (Structural Query)
    if state.cypher_result:
        for res in state.cypher_result:
            content = _flatten_dict(res)
            raw_akus.append({
                "content": f"Database Fact: {content}",
                "origin": "Graph DB",
                "method": "Cypher Query",
                "metadata": {"type": "Query Result", "name": "Database Result"}
            })

    # Deduplication: Merge identical content
    final_akus = []
    seen_content = {} # content -> list_index
    
    for aku in raw_akus:
        content = aku["content"]
        if content in seen_content:
            existing = final_akus[seen_content[content]]
            # Merge methods if different
            if aku["method"] not in existing["method"]:
                existing["method"] = f"{existing['method']} & {aku['method']}"
        else:
            seen_content[content] = len(final_akus)
            final_akus.append(aku)

    # Assign sequential indices [1], [2], ...
    for i, aku in enumerate(final_akus, 1):
        aku["index"] = i

    return final_akus


def _format_akus_for_prompt(akus: list[dict]) -> str:
    """Format AKUs into a readable block for the prompt.

    Args:
        akus: List of indexed AKU dicts.

    Returns:
        str: Formatted context block.
    """
    if not akus:
        return "No context available."
    
    lines = []
    for aku in akus:
        lines.append(
            f"[{aku['index']}] {aku['content']} "
            f"(Origin: {aku['origin']} | Method: {aku['method']})"
        )
    return "\n".join(lines)


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


async def planner(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Analyze query and generate a structured retrieval and reasoning plan.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: State update with strategy, plan, target_entity_types, and expected_outcome.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = _get_query_text(last_message)

    schema = format_schema_for_prompt(Path(settings.data_volume_path) / "graph_schema.json")
    prompt = PLANNER_PROMPT.format(schema=schema, query=query_text)

    response_text = await asyncio.to_thread(
        gemini_generate, client, prompt, model=configuration.model
    )
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        import json
        data = json.loads(response_text)
        strategy = data.get("strategy", "hybrid").lower()
        is_fast_track = data.get("is_fast_track", False)
        plan = data.get("plan", "")
        target_entity_types = data.get("target_entity_types", [])
        expected_outcome = data.get("expected_outcome", "")
    except Exception:
        logger.warning("planner_parse_failed", raw=response_text)
        strategy = "hybrid"
        is_fast_track = False
        plan = "Fallback to hybrid search."
        target_entity_types = []
        expected_outcome = ""

    if strategy not in ("local", "global", "drift", "hybrid", "structural"):
        strategy = "hybrid"

    logger.info("planner_done", strategy=strategy, fast_track=is_fast_track, plan=plan)
    return {
        "strategy": strategy, 
        "is_fast_track": is_fast_track,
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
    client = await gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = _get_query_text(last_message)

    schema_path = Path(settings.data_volume_path) / "graph_schema.json"
    schema = format_schema_for_prompt(schema_path)
    prompt = STRATEGY_PROMPT.format(schema=schema, query=query_text)

    # Use JSON response if supported or strip markdown
    response_text = await asyncio.to_thread(
        gemini_generate, client, prompt, model=configuration.model
    )
    response_text = response_text.strip()
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
    client = await gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = _get_query_text(last_message)

    schema = format_schema_for_prompt(Path(settings.data_volume_path) / "graph_schema.json")
    prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, query=query_text)

    response_text = await asyncio.to_thread(
        gemini_generate, client, prompt, model=configuration.model
    )
    response_text = response_text.strip()
    
    # Clean up markdown code blocks
    if "```cypher" in response_text:
        response_text = response_text.split("```cypher")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    logger.info("cypher_generator_done", query=response_text)
    return {"generated_cypher": response_text}


import re

def _resolve_aku_legend(answer: str, akus: list[dict], source_urls: dict[str, dict[str, str]], llm_evidence: dict[str, str] = None) -> tuple[str, str]:
    """Build a structured legend and re-index citations sequentially.

    Parses the answer for [N] citations, maps them to [1], [2], etc. based
    on appearance order, and resolves them to their metadata.

    Args:
        answer: The LLM-generated answer text.
        akus: The list of Atomic Knowledge Units provided to the LLM.
        source_urls: Mapping of QID/article_id to metadata (name, url).
        llm_evidence: Optional mapping of raw index to verbatim quote or summary from LLM.

    Returns:
        tuple[str, str]: (updated_answer, legend_section)
    """
    # 1. Find all raw indices in order of appearance
    raw_citations = re.findall(r"\[(\d+)\]", answer)
    if not raw_citations:
        return answer, ""

    # 2. Map raw index -> new sequential index
    raw_to_new = {}
    new_idx = 1
    for raw in raw_citations:
        if raw not in raw_to_new:
            raw_to_new[raw] = str(new_idx)
            new_idx += 1

    # 3. Update the answer text with new indices
    # We use a lambda to avoid replacing partial matches
    updated_answer = re.sub(r"\[(\d+)\]", lambda m: f"[{raw_to_new[m.group(1)]}]", answer)

    # 4. Build the legend
    aku_map = {str(aku["index"]): aku for aku in akus}
    legend_lines = ["\n---", "**Sources & Evidence Path:**"]
    
    # Iterate through our mapping to build the legend in order of appearance
    for raw, seq in raw_to_new.items():
        aku = aku_map.get(raw)
        if not aku:
            continue
        
        metadata = aku.get("metadata", {})
        qid = metadata.get("qid") or metadata.get("article_id")
        source_info = source_urls.get(qid) if qid else None
        
        origin = aku.get("origin", "Unknown")
        method = aku.get("method", "Unknown")
        
        # Priority 1: Use LLM-provided verbatim quote/summary if available
        content_label = llm_evidence.get(raw) if llm_evidence else None
        
        line = f"- `[{seq}]` "
        
        # Format based on origin
        if origin == "Vector DB":
            # If no LLM quote, fallback to cleaned content snippet
            if not content_label:
                content_label = aku['content'].split(": ", 1)[-1] if ": " in aku['content'] else aku['content']
                content_label = f"{content_label[:120]}..."
            
            line += f'"{content_label}"'
            
            # Add Chunk ID for surgical auditing
            if metadata.get("chunk_id"):
                line += f" | ID: {metadata['chunk_id']}"

            if source_info:
                name = source_info.get("name", "Source")
                url = source_info.get("wikipedia_url")
                if url:
                    line += f" ([{name}]({url}))"
        else:
            # Graph DB or other structural sources
            if content_label:
                line += f"{content_label}"
            elif metadata.get("name"):
                line += f"{metadata['name']}"
            else:
                line += f"{aku['content'][:120]}..."

            if source_info:
                name = source_info.get("name", "Source")
                url = source_info.get("wikipedia_url")
                if url:
                    line += f" ([{name}]({url}))"

        line += f" | Origin: {origin} | Method: {method}"
        legend_lines.append(line)

    return updated_answer, "\n".join(legend_lines)


def _check_faithfulness(answer: str, akus: list[dict]) -> dict[str, Any]:
    """Heuristic check for attribution density and potential drift.

    Args:
        answer: The generated answer text.
        akus: The provided context units.

    Returns:
        dict[str, Any]: Results of the check (is_faithful, issues).
    """
    citations = re.findall(r"\[(\d+)\]", answer)
    unique_citations = set(citations)
    
    # 1. Density check: At least 1 citation per 100 characters for non-trivial answers
    if len(answer) > 200 and len(citations) < (len(answer) / 200):
        return {"is_faithful": False, "issue": "Low citation density"}

    # 2. Hallucination check: Are all cited indices in the context?
    valid_indices = {str(aku["index"]) for aku in akus}
    hallucinated = unique_citations - valid_indices
    if hallucinated:
        return {"is_faithful": False, "issue": f"Hallucinated indices: {hallucinated}"}

    return {"is_faithful": True, "issue": None}


async def synthesize_answer(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Synthesize a final answer from all retrieved context using AKUs.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with AIMessage answer and akus.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = _get_query_text(last_message)

    # Homogenize context into AKUs
    akus = homogenize_context(state)

    # Build structured prompt with unified AKUs
    schema_path = Path(settings.data_volume_path) / "graph_schema.json"
    schema = format_schema_for_prompt(schema_path)
    prompt = SYNTHESIS_PROMPT.format(
        schema=schema,
        akus=_format_akus_for_prompt(akus),
        critique=state.critique,
        guide=state.retrieval_guide,
        query=query_text,
    )
    raw_response = await asyncio.to_thread(
        gemini_generate, client, prompt, model=configuration.model
    )
    raw_response = raw_response.strip()
    
    # Clean up JSON if LLM added markdown blocks
    if raw_response.startswith("```json"):
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
    elif raw_response.startswith("```"):
        raw_response = raw_response.replace("```", "").strip()

    try:
        import json
        data = json.loads(raw_response)
        answer = data.get("answer", "")
        evidence = data.get("evidence", {})
    except Exception:
        logger.warning("synthesize_answer_json_parse_failed", raw=raw_response)
        # Fallback to assuming the raw response is the answer if parsing fails
        answer = raw_response
        evidence = {}

    # Phase 3: Streamlined Resolution & Semantic Validation
    faithfulness = _check_faithfulness(answer, akus)
    if not faithfulness["is_faithful"]:
        logger.warning("synthesis_faithfulness_low", issue=faithfulness["issue"])

    # Build the structural attribution report (Final Mapping)
    updated_answer, legend = _resolve_aku_legend(answer, akus, state.source_urls, llm_evidence=evidence)
    final_answer = updated_answer + legend

    logger.info("synthesize_answer_done", length=len(final_answer), aku_count=len(akus), faithful=faithfulness["is_faithful"])
    return {"messages": [AIMessage(content=final_answer)], "akus": akus}
