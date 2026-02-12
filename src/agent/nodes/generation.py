"""Generation nodes for the LangGraph agent.

LLM-powered nodes for query analysis and answer synthesis.
"""

from typing import Any

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.infrastructure.clients import gemini_client
from agent.state import State
from agent.tools.gemini import gemini_generate

logger = structlog.get_logger()

STRATEGY_PROMPT = (
    "You are a query classifier for a knowledge graph search system.\n"
    "Classify the following user query into exactly one strategy:\n"
    '- "local": The query asks about specific entities, people, places, '
    "or concrete facts.\n"
    '- "global": The query asks about broad themes, trends, summaries, '
    "or community-level patterns.\n"
    '- "hybrid": The query combines specific entity questions with '
    "broader thematic context.\n\n"
    "User query: {query}\n\n"
    "Respond with ONLY one word: local, global, or hybrid."
)

SYNTHESIS_PROMPT = (
    "You are a knowledgeable assistant answering questions using a "
    "knowledge graph.\n"
    "Use the following retrieved context to answer the user's question.\n"
    "Cite sources using [Source Name](URL) format when available.\n\n"
    "{context}\n\n"
    "User question: {query}\n\n"
    "Provide a comprehensive, well-structured answer based on the "
    "context above. If the context does not contain enough information, "
    "say so clearly."
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
        summary = r.get("summary", "No summary")
        lines.append(f"- Community {cid}: {summary}")
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
        lines.append(f"- [score={score:.3f}] {text}")
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


async def query_analyzer(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user query into a search strategy.

    Calls Gemini to determine whether the query requires local, global,
    or hybrid search.

    Args:
        state: Current graph state.
        config: LangGraph runtime configuration.

    Returns:
        dict[str, Any]: Partial state update with strategy.
    """
    configuration = Configuration.from_runnable_config(config)
    client = gemini_client.get_client()
    last_message = state.messages[-1]
    query_text = last_message.content

    prompt = STRATEGY_PROMPT.format(query=query_text)
    response = gemini_generate(client, prompt, model=configuration.model)
    strategy = response.strip().lower()

    # Validate to one of the three strategies
    if strategy not in ("local", "global", "hybrid"):
        logger.warning(
            "query_analyzer_fallback",
            raw=strategy,
            fallback="hybrid",
        )
        strategy = "hybrid"

    logger.info("query_analyzer_done", strategy=strategy)
    return {"strategy": strategy}


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

    # Build context from all retrieval results
    sections = [
        _format_entities(state.entities),
        _format_relationships(state.relationships),
        _format_community_reports(state.community_reports),
        _format_chunks(state.chunk_evidence),
        _format_sources(state.source_urls),
    ]
    context = "\n\n".join(s for s in sections if s)

    prompt = SYNTHESIS_PROMPT.format(context=context, query=query_text)
    answer = gemini_generate(client, prompt, model=configuration.model)

    logger.info("synthesize_answer_done", length=len(answer))
    return {"messages": [AIMessage(content=answer)]}
