# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for the embedding processing tools.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for the embedding processing tools."""

from agent.tools.embeddings import process_embedding

def test_process_embedding_normalization():
    """Verify that embeddings are L2 normalized."""
    raw = [3.0, 4.0] # Magnitude = 5.0
    result = process_embedding(raw)
    
    assert result == [0.6, 0.8]
    # Check magnitude is 1.0
    assert sum(x*x for x in result) == pytest.approx(1.0)

import pytest
