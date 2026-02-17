# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tool for synthesizing answers, validating citations, and formatting output.
# Encapsulates logic for processing AKUs (Atomic Knowledge Units), calculating
# importance scores, checking faithfulness, and generating the final cited answer with a legend.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

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
    
    Handles both single [1] and grouped [1, 2] citations.
    
    Args:
        answer: The generated answer text.
        akus: List of AKU dictionaries.
        source_urls: Mapping of QIDs to source info.
        llm_evidence: Optional explanation mapping.
        
    Returns:
        tuple[str, str]: The updated answer and legend string.
    """
    # 1. Find citation blocks like [1], [1, 2], [1, 3, 5]
    # Only match 1-2 digit numbers to avoid capturing years like [2024]
    _CITE_NUM = r"\d{1,2}"
    citation_pattern = re.compile(
        rf"\[({_CITE_NUM}(?:\s*,\s*{_CITE_NUM})*)\]"
    )

    raw_to_new: dict[str, str] = {}
    new_idx = 1
    citation_order: list[str] = []

    def replacement_handler(match: re.Match) -> str:
        nonlocal new_idx
        content = match.group(1)
        raw_ids = [s.strip() for s in content.split(",") if s.strip().isdigit()]

        if not raw_ids:
            return match.group(0)

        new_ids = []
        for raw in raw_ids:
            if raw not in raw_to_new:
                raw_to_new[raw] = str(new_idx)
                new_idx += 1
                citation_order.append(raw)
            new_ids.append(raw_to_new[raw])

        return f"[{', '.join(new_ids)}]"

    updated_answer = citation_pattern.sub(replacement_handler, answer)
    
    if not raw_to_new:
        return answer, ""

    # 2. Build the legend
    aku_map = {str(aku["index"]): aku for aku in akus}
    legend_lines = ["\n---", "**Sources & Evidence Path:**"]
    
    for idx, raw in enumerate(citation_order, 1):
        seq = str(idx)
        aku = aku_map.get(raw)
        if not aku: continue
        
        metadata = aku.get("metadata", {})
        qid = metadata.get("qid") or metadata.get("article_id")
        source_info = source_urls.get(qid) if qid and source_urls else None
        
        origin = aku.get("origin", "Unknown")
        method = aku.get("method", "Unknown")
        
        # Determine the label
        content_label = ""
        if llm_evidence and raw in llm_evidence:
            content_label = llm_evidence[raw]
        
        # Build the line
        line_parts = [f"- `[{seq}]`"]
        
        if origin == "Vector DB":
            # For text chunks, prefer the explanation or a snippet
            if not content_label:
                snippet = aku['content']
                if ": " in snippet:
                    snippet = snippet.split(": ", 1)[-1]
                content_label = f"{snippet[:120]}..." if len(snippet) > 120 else snippet
            line_parts.append(f'"{content_label}"')
        else:
            # For Graph DB, show the structural name (Triple) AND the explanation if available
            name = metadata.get("name")
            if not name:
                 name = aku['content'][:120] + "..."
            
            if content_label:
                line_parts.append(f"{name} ({content_label})")
            else:
                line_parts.append(f"{name}")

        if source_info:
            name = source_info.get("name", "Source")
            url = source_info.get("wikipedia_url")
            if url:
                 line_parts.append(f"[{content_label}]({url})")
            else:
                 line_parts.append(f"{content_label}")
        else:
            line_parts.append(f"{content_label}")

        if metadata.get("chunk_id"):
             line_parts.append(f"| ID: {metadata['chunk_id']}")
        
        line_parts.append(f"| Origin: {origin} | Method: {method}")
        legend_lines.append(" ".join(line_parts))

    return updated_answer, "\n".join(legend_lines)


def check_faithfulness(answer: str, akus: list[dict]) -> dict[str, str | bool | None]:
    """Heuristic check for attribution density.
    
    Args:
        answer: The generated answer text.
        akus: List of AKU dictionaries.
        
    Returns:
        dict: Result containing 'is_faithful' and 'issue'.
    """
    citations = re.findall(r"\[(\d{1,2})\]", answer)
    if len(answer) > 200 and len(citations) < (len(answer) / 200):
        return {"is_faithful": False, "issue": "Low citation density"}
    
    valid_indices = {str(aku["index"]) for aku in akus}
    hallucinated = set(citations) - valid_indices
    
    if hallucinated: 
        return {"is_faithful": False, "issue": f"Hallucinated indices: {hallucinated}"}
    
    return {"is_faithful": True, "issue": None}
