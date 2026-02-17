# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Generation nodes for the LangGraph agent.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Generation nodes for the LangGraph agent.

LLM-powered nodes for query analysis and answer synthesis.
"""

import asyncio
import json
from typing import Any

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client
from agent.nodes.models import RouterResponse, SynthesisResponse
from agent.state import State
from agent.tools.gemini import gemini_generate
from agent.tools.synthesis import (
    calculate_aku_importance,
    check_faithfulness,
    flatten_dict,
    resolve_aku_legend,
)

logger = structlog.get_logger()

ROUTER_PROMPT = (
    "You are the Intelligent Router for a music-domain GraphRAG agent.\n\n"
    "Your goal is to analyze the user query and determine the most efficient retrieval strategy.\n\n"
    "STRATEGIES:\n"
    '- "local": Specific entities (PERSON, GROUP, ALBUM, CITY) and their direct properties.\n'
    '- "global": Broad thematic questions, SOCIAL_MOVEMENT, GENRE, or trends spanning many entities.\n'
    '- "drift": Multi-hop reasoning, following chains of influence, or abstract "impact" questions.\n'
    '- "structural": Quantitative queries: counts ("How many"), lists ("List all"), or specific property filters ("released in 1989").\n'
    '- "hybrid": Complex queries requiring both thematic context and specific database facts.\n\n'
    "CRITICAL DISCRIMINATION RULES:\n"
    "1. DO NOT use 'structural' for abstract queries using words like 'impact', 'influence', 'legacy', 'development', or 'significance'. These are better handled by 'drift' or 'global'.\n"
    "2. ONLY use 'structural' if the question can be answered by a discrete relationship in the schema.\n"
    "3. Set 'is_fast_track' to true ONLY for simple, single-entity factoid lookups.\n\n"
    "User query: {query}"
)

SYNTHESIS_PROMPT = (
    "You are a knowledgeable music assistant answering questions using a "
    "combination of community thematic summaries, knowledge graph facts, "
    "original source text, and the underlying graph schema.\n\n"
    "UNIFIED CONTEXT (Atomic Knowledge Units):\n"
    "{akus}\n\n"
    "RETRIEVAL GUIDE (Instructions for focus):\n"
    "{guide}\n\n"
    "Using the above, answer the user's question: {query}\n\n"
    "Guidelines:\n"
    "1. ATTENTION PINNING: Every claim MUST be immediately followed by the relevant index(es) in brackets (e.g., [1] or [1][3]).\n"
    "2. CONSOLIDATION: If you have a long list of similar facts, summarize them and cite a representative sample (max 5 indices).\n"
    "3. TRUTH HIERARCHY: Prioritize AKUs from 'Graph DB' for structural facts and 'Vector DB' for descriptive context.\n"
    "4. VERBATIM EVIDENCE: For every 'Vector DB' AKU you cite, you MUST provide a verbatim quote from its text to prove the claim.\n"
    "5. CLEAN SYNTHESIS: Provide the answer in plain text without markdown headers or links."
)


def homogenize_context(state: State) -> list[dict]:
    """Unify and prune retrieval results into a focused list of AKUs.
    
    Args:
        state: Current graph state containing retrieval results.
        
    Returns:
        list[dict]: List of homogenized Atomic Knowledge Units.
    """
    raw_akus = []

    # 1. Process Entities
    for e in state.entities:
        name = e.get("name", "Unknown")
        desc = e.get("description", "No description")
        e_type = e.get("type", "Entity")
        raw_akus.append({
            "content": f"Entity ({e_type}): {name} - {desc}",
            "origin": e.get("origin", "Graph DB"),
            "method": e.get("method", "Entity Search"),
            "raw_relevance_score": e.get("score", 0.5),
            "metadata": {
                "qid": e.get("qid"), "type": e_type, "name": name, 
                "mention_count": e.get("mention_count", 0), "pagerank": e.get("pagerank", 0)
            }
        })

    # 2. Process Relationships
    for r in state.relationships:
        source = r.get("source_name", "?")
        rel = r.get("relationship", "RELATED_TO")
        target = r.get("target_name") or r.get("name", "?")
        desc = r.get("rel_description", "")
        content = f"Relationship: {source} --[{rel}]--> {target}"
        if desc: content += f" ({desc})"
        raw_akus.append({
            "content": content,
            "origin": r.get("origin", "Graph DB"),
            "method": r.get("method", "Neighborhood Expansion"),
            "raw_relevance_score": r.get("score", 0.5),
            "metadata": {"qid": r.get("qid"), "type": rel, "mention_count": r.get("mention_count", 0)}
        })

    # 3. Process Text Chunks
    for c in state.chunk_evidence:
        content = c.get("text", "")
        raw_akus.append({
            "content": f"Text Evidence: {content}",
            "origin": c.get("origin", "Vector DB"),
            "method": c.get("method", "Surgical Search"),
            "raw_relevance_score": c.get("raw_score", c.get("score", 0.5)),
            "metadata": {"chunk_id": c.get("id"), "type": "Text Chunk", "mention_count": c.get("mention_count", 0)}
        })

    # 4. Process Community Reports
    for r in state.community_reports:
        cid = r.get("community_id", "?")
        title = r.get("title", "")
        summary = r.get("summary", "No summary")
        raw_akus.append({
            "content": f"Thematic Summary (Community {cid}): {title} - {summary}",
            "origin": r.get("origin", "Vector DB"),
            "method": r.get("method", "Community Search"),
            "raw_relevance_score": 0.9, # High weight for summaries
            "metadata": {"community_id": cid, "type": "Community Report"}
        })

    # 5. Process Cypher Results
    for res in state.cypher_result:
        content = flatten_dict(res)
        raw_akus.append({
            "content": f"Database Fact: {content}",
            "origin": "Graph DB",
            "method": "Cypher Query",
            "raw_relevance_score": 1.0,
            "metadata": {"type": "Query Result"}
        })

    # Deduplication and Scoring
    unique_akus = []
    seen_content = {}
    for aku in raw_akus:
        content = aku["content"]
        if content in seen_content:
            existing = unique_akus[seen_content[content]]
            if aku["method"] not in existing["method"]:
                existing["method"] = f"{existing['method']} & {aku['method']}"
        else:
            seen_content[content] = len(unique_akus)
            aku["importance"] = calculate_aku_importance(aku)
            unique_akus.append(aku)

    # Pruning: Sort by importance and cap redundancy
    unique_akus.sort(key=lambda x: x["importance"], reverse=True)
    
    final_akus = []
    structural_count = 0
    MAX_AKUS, MAX_STRUCTURAL = 12, 5
    
    for aku in unique_akus:
        if len(final_akus) >= MAX_AKUS: break
        
        is_structural = "Database Fact" in aku["content"] or "Relationship" in aku["content"]
        if is_structural:
            if structural_count >= MAX_STRUCTURAL: continue
            structural_count += 1
            
        final_akus.append(aku)

    for i, aku in enumerate(final_akus, 1):
        aku["index"] = i
    return final_akus


def _format_akus_for_prompt(akus: list[dict]) -> str:
    """Format AKUs into a readable block for the prompt.
    
    Args:
        akus: List of Atomic Knowledge Units.
        
    Returns:
        str: Formatted string for LLM consumption.
    """
    if not akus: return "No context available."
    return "\n".join([f"[{aku['index']}] {aku['content']} (Origin: {aku['origin']} | Method: {aku['method']})" for aku in akus])


def _get_query_text(message: Any) -> str:
    """Extract string content from a message.
    
    Args:
        message: LangChain message object.
        
    Returns:
        str: Extracted text content.
    """
    content = message.content
    if isinstance(content, str): return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text": return block.get("text", "")
            if isinstance(block, str): return block
    return str(content)


async def router(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Consolidated router node with structured output and context caching.
    
    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.
        
    Returns:
        dict[str, Any]: Partial state update with strategy and plan.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    query_text = _get_query_text(state.messages[-1])
    
    # Context Caching: Use schema from client's pseudo-cache
    system_instruction = gemini_client.get_schema_instruction()
    prompt = ROUTER_PROMPT.format(query=query_text)

    response = await asyncio.to_thread(
        gemini_generate, 
        client, 
        prompt, 
        model=configuration.model,
        response_schema=RouterResponse,
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )

    if isinstance(response, RouterResponse):
        strategy = response.strategy.lower()
        is_fast_track = response.is_fast_track
        target_entity_types = response.target_entity_types
        plan = response.plan
    elif isinstance(response, dict):
        strategy = response.get("strategy", "hybrid").lower()
        is_fast_track = response.get("is_fast_track", False)
        target_entity_types = response.get("target_entity_types", [])
        plan = response.get("plan", "Optimized routing.")
    else:
        logger.warning("router_parse_failed", raw=response)
        strategy, is_fast_track, target_entity_types, plan = "hybrid", False, [], "Fallback."

    # Validate strategy
    if strategy not in ("local", "global", "drift", "hybrid", "structural"):
        strategy = "hybrid"

    logger.info("router_done", strategy=strategy, fast_track=is_fast_track)
    return {"strategy": strategy, "is_fast_track": is_fast_track, "target_entity_types": target_entity_types, "plan": plan, "iteration_count": 0}


async def synthesize_answer(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Synthesize a final answer from all retrieved context using AKUs.
    
    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.
        
    Returns:
        dict[str, Any]: Partial state update with the final answer message.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    query_text = _get_query_text(state.messages[-1])
    akus = homogenize_context(state)
    
    system_instruction = gemini_client.get_schema_instruction()
    prompt = SYNTHESIS_PROMPT.format(akus=_format_akus_for_prompt(akus), guide=state.retrieval_guide, query=query_text)
    
    response = await asyncio.to_thread(
        gemini_generate, 
        client, 
        prompt, 
        model=configuration.model,
        response_schema=SynthesisResponse,
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )

    if isinstance(response, SynthesisResponse):
        answer = response.answer
        evidence = {str(item.index): item.content for item in response.evidence}
    elif isinstance(response, dict):
        answer = response.get("answer", "")
        evidence_list = response.get("evidence", [])
        evidence = {str(item.get("index")): item.get("content") for item in evidence_list if isinstance(item, dict)}
    else:
        answer, evidence = str(response), {}

    _ = check_faithfulness(answer, akus) # Result logged or used if needed
    updated_answer, legend = resolve_aku_legend(answer, akus, state.source_urls, llm_evidence=evidence)
    final_answer = updated_answer + legend
    logger.info("synthesize_answer_done", length=len(final_answer), aku_count=len(akus))
    return {"messages": [AIMessage(content=final_answer)], "akus": akus}
