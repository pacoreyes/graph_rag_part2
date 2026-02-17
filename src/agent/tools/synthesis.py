# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tool for synthesizing answers, validating citations, and formatting output.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tool for synthesizing answers, validating citations, and formatting output.

Encapsulates logic for processing AKUs (Atomic Knowledge Units), calculating
importance scores, checking faithfulness, and generating the final cited answer with a legend.
"""

import math
import re


def flatten_dict(data: dict) -> str:
    """Flatten a dictionary into a comma-separated string of key-values.
    
    Args:
        data: Dictionary to flatten.
        
    Returns:
        str: Comma-separated string representation.
    """
    parts = []
    for k, v in data.items():
        if v is not None:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)


def calculate_aku_importance(aku: dict) -> float:
    """Calculate a base importance score for an AKU.
    
    Args:
        aku: The Atomic Knowledge Unit dictionary.
        
    Returns:
        float: Calculated importance score.
    """
    metadata = aku.get("metadata", {})
    # Use mention_count (log-scaled) as primary importance signal
    mention_count = float(metadata.get("mention_count") or 0)
    pagerank = float(metadata.get("pagerank") or 0)
    relevance_val = aku.get("raw_relevance_score")
    relevance = float(relevance_val if relevance_val is not None else 0.5)
    
    authority = math.log1p(mention_count) + math.log1p(pagerank)
    return relevance * (1 + authority)


def resolve_aku_legend(
    answer: str, 
    akus: list[dict], 
    source_urls: dict[str, dict[str, str]], 
    llm_evidence: dict[str, str] = None
) -> tuple[str, str]:
    """Build a structured legend and re-index citations sequentially.
    
    Args:
        answer: The generated answer text containing raw citations like [12].
        akus: List of AKU dictionaries used for context.
        source_urls: Dictionary mapping QIDs to source info.
        llm_evidence: Optional dictionary mapping indices to LLM-generated evidence summaries.
        
    Returns:
        tuple[str, str]: The updated answer with re-indexed citations, and the legend string.
    """
    raw_citations = re.findall(r"\[(\d+)\]", answer)
    if not raw_citations: return answer, ""

    raw_to_new, new_idx = {}, 1
    # We use a dict to preserve order and uniqueness of citations found in text
    citation_order = []
    for raw in raw_citations:
        if raw not in raw_to_new:
            raw_to_new[raw] = str(new_idx)
            new_idx += 1
            citation_order.append(raw)

    # Replace citations in the text
    updated_answer = re.sub(r"\[(\d+)\]", lambda m: f"[{raw_to_new[m.group(1)]}]", answer)
    
    aku_map = {str(aku["index"]): aku for aku in akus}
    legend_lines = ["---", "**Sources & Evidence Path:**"]
    
    # Build legend based on the order they appear (re-indexed 1, 2, 3...)
    # The 'raw_to_new' values are '1', '2', '3'... so we can just iterate 1..new_idx-1
    # But we need to map back to the original raw ID to find the AKU.
    # We can iterate through our 'citation_order' which has the raw IDs in appearance order.
    
    for idx, raw in enumerate(citation_order, 1):
        seq = str(idx) # This matches raw_to_new[raw]
        aku = aku_map.get(raw)
        if not aku: continue
        
        metadata = aku.get("metadata", {})
        qid = metadata.get("qid") or metadata.get("article_id")
        # In a real graph, we'd look up the URL. For now, we use a placeholder or check source_urls.
        source_info = source_urls.get(qid) if qid and source_urls else None
        
        origin = aku.get("origin", "Unknown")
        method = aku.get("method", "Unknown")
        
        # Determine the label for the legend entry
        content_label = ""
        if llm_evidence and raw in llm_evidence:
            content_label = llm_evidence[raw]
        elif metadata.get("name"):
            content_label = metadata["name"]
        else:
            # Fallback to a snippet of content
            snippet = aku['content']
            if ": " in snippet:
                snippet = snippet.split(": ", 1)[-1]
            content_label = f"{snippet[:120]}..." if len(snippet) > 120 else snippet

        # Build the line
        line_parts = [f"- `[{seq}]`"]
        
        # Add link if available
        if source_info:
            name = source_info.get("name", "Source")
            url = source_info.get("wikipedia_url")
            if url:
                 line_parts.append(f"[{content_label}]({url})")
            else:
                 line_parts.append(f"{content_label}")
        else:
            line_parts.append(f"{content_label}")

        # Add metadata
        if metadata.get("chunk_id"):
             line_parts.append(f"| ID: {metadata['chunk_id']}")
        
        line_parts.append(f"| Origin: {origin} | Method: {method}")
        
        legend_lines.append(" ".join(line_parts))

    return updated_answer, "".join(legend_lines)


def check_faithfulness(answer: str, akus: list[dict]) -> dict[str, str | bool | None]:
    """Heuristic check for attribution density.
    
    Args:
        answer: The generated answer text.
        akus: List of AKU dictionaries.
        
    Returns:
        dict: Result containing 'is_faithful' and 'issue'.
    """
    citations = re.findall(r"\[(\d+)\]", answer)
    if len(answer) > 200 and len(citations) < (len(answer) / 200):
        return {"is_faithful": False, "issue": "Low citation density"}
    
    valid_indices = {str(aku["index"]) for aku in akus}
    hallucinated = set(citations) - valid_indices
    
    if hallucinated: 
        return {"is_faithful": False, "issue": f"Hallucinated indices: {hallucinated}"}
    
    return {"is_faithful": True, "issue": None}
