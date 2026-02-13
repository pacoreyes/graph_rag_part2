"""Critic node for self-correction and answer verification."""

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client
from agent.state import State
from agent.tools.gemini import gemini_generate

logger = structlog.get_logger()

CRITIC_PROMPT = """You are a Quality Control Auditor for a music GraphRAG agent.
Your task is to compare the FINAL ANSWER against the RETRIEVED DATA and the ORIGINAL PLAN to ensure accuracy and completeness.

ORIGINAL PLAN & EXPECTED OUTCOME:
{plan}

RETRIEVED DATA SUMMARY:
- Entities found: {entity_count}
- Cypher results found: {cypher_count}
- Community reports used: {community_count}
- Text chunks used: {chunk_count}

FINAL ANSWER:
{answer}

CRITERIA:
1. COMPLETENESS: Does the answer include all specific items found in the data (e.g., if Cypher found 10 bands, are all 10 in the answer)?
2. ACCURACY: Does the answer contradict any of the retrieved facts?
3. HALLUCINATION: Does the answer include facts not present in the context?

JSON RESPONSE FORMAT:
{{
  "verdict": "pass" or "fail",
  "critique": "Detailed explanation of what is missing or incorrect. Be specific.",
  "suggestions": "Instructions for the synthesizer to fix the answer."
}}

Respond with ONLY the JSON object."""


async def answer_critic(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate the generated answer and decide if it needs refinement.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: State update with critique and incremented iteration_count.
    """
    configuration = Configuration.from_runnable_config(config)
    client = gemini_client.get_client()
    
    # Get the last AI message (the answer)
    answer = ""
    for msg in reversed(state.messages):
        if msg.type == "ai":
            answer = msg.content
            break

    if not answer:
        return {"critique": "No answer found to evaluate.", "iteration_count": state.iteration_count + 1}

    prompt = CRITIC_PROMPT.format(
        plan=state.plan,
        entity_count=len(state.entities),
        cypher_count=len(state.cypher_result),
        community_count=len(state.community_reports),
        chunk_count=len(state.chunk_evidence),
        answer=answer
    )

    response_text = gemini_generate(client, prompt, model=configuration.model).strip()
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        import json
        data = json.loads(response_text)
        verdict = data.get("verdict", "pass").lower()
        critique = data.get("critique", "")
        suggestions = data.get("suggestions", "")
    except Exception:
        logger.warning("critic_parse_failed", raw=response_text)
        verdict = "pass"
        critique = ""
        suggestions = ""

    full_critique = f"VERDICT: {verdict}\nCRITIQUE: {critique}\nSUGGESTIONS: {suggestions}"
    logger.info("answer_critic_done", verdict=verdict, iteration=state.iteration_count)

    return {
        "critique": full_critique if verdict == "fail" else "",
        "iteration_count": state.iteration_count + 1
    }
