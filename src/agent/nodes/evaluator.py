"""Evaluator node to assess retrieval quality and prune context noise."""

import asyncio
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client
from agent.state import State
from agent.tools.gemini import gemini_generate

logger = structlog.get_logger()

EVALUATOR_PROMPT = """You are the Lead Retrieval Auditor for a music GraphRAG agent.
Your task is to orchestrate multiple retrieval paths (Structural, Semantic, Thematic) and decide if the current evidence is sufficient.

USER QUERY: {query}
ORIGINAL PLAN: {plan}

COLLECTED DATA:
- Entities (Semantic/Local Path): {entity_names}
- Cypher Results (Structural Path): {cypher_results}
- Community Reports (Global/Thematic Path): {community_count}

GUIDELINES:
1. TRIANGULATION: Do not trust a single source. Even if Cypher found 10 items, are they clearly verified by the entity descriptions or community reports?
2. COMPLEXITY CHECK: If the query involves multiple constraints (e.g., '10 bands' AND 'Techno' AND 'UK'), the chance of error is higher. MARK AS 'insufficient' to keep the deep text search active as a safety net.
3. SUFFICIENCY: Only mark as 'sufficient' if the data is high-precision, relevant, and consistent across multiple paths.
4. SYNTHESIS GUIDE: Provide a strategy for the writer. If Cypher results look noisy (e.g., mismatched countries), tell the writer to filter them using text evidence.

JSON RESPONSE FORMAT:
{{
  "status": "sufficient" or "insufficient",
  "reasoning": "Explain the agreement or conflict between retrieval paths.",
  "synthesis_guide": "Specific instructions on which path to trust most and how to cross-reference (e.g., 'Use Cypher results as candidates, but verify their origin using the text chunks.')"
}}

Respond with ONLY the JSON object."""


async def retrieval_evaluator(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Assess if the current retrieval is enough or if we need deep search (chunks).

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: State update with retrieval_guide and skip_deep_search flag.
    """
    configuration = Configuration.from_runnable_config(config)
    client = await gemini_client.get_client()
    query_text = state.messages[-1].content

    # Summarize data for the prompt
    entity_names = [e.get("name", "Unknown") for e in state.entities[:15]]
    
    prompt = EVALUATOR_PROMPT.format(
        query=query_text,
        plan=state.plan,
        entity_names=str(entity_names),
        cypher_results=str(state.cypher_result[:15]),
        community_count=len(state.community_reports)
    )

    response_text = await asyncio.to_thread(
        gemini_generate, client, prompt, model=configuration.model
    )
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        import json
        data = json.loads(response_text)
        status = data.get("status", "insufficient").lower()
        reasoning = data.get("reasoning", "")
        guide = data.get("synthesis_guide", "")
    except Exception:
        logger.warning("evaluator_parse_failed", raw=response_text)
        status = "insufficient"
        reasoning = "Failed to parse evaluation. Defaulting to deep search."
        guide = ""

    skip = (status == "sufficient")
    logger.info("retrieval_evaluator_done", status=status, skip_deep=skip, reasoning=reasoning)

    return {
        "retrieval_guide": guide,
        "skip_deep_search": skip
    }
