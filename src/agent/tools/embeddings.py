# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Reusable embedding processing functions.
# Pure functions without heavy dependencies like torch or transformers.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import math


def process_embedding(
    embedding: list[float],
) -> list[float]:
    """Process a raw embedding vector.

    Handles L2 normalization in pure Python.

    Args:
        embedding: Raw embedding vector from the model/API.

    Returns:
        list[float]: The L2-normalized embedding vector.
    """
    # L2 normalize: v / sqrt(sum(x^2 for x in v))
    squared_sum = sum(x * x for x in embedding)
    magnitude = math.sqrt(max(squared_sum, 1e-9))
    
    return [x / magnitude for x in embedding]
