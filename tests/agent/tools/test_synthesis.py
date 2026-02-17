# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Tests for synthesis tools (citation handling, faithfulness).
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Tests for synthesis tools (citation handling, faithfulness)."""

from agent.tools.synthesis import check_faithfulness, resolve_aku_legend


def _make_akus(n: int) -> list[dict]:
    """Create n minimal AKU dicts with sequential indices."""
    return [
        {
            "index": i,
            "content": f"AKU content {i}",
            "origin": "Graph DB",
            "method": "Test",
            "metadata": {"name": f"Entity {i}"},
        }
        for i in range(1, n + 1)
    ]


def test_resolve_aku_legend_ignores_years():
    """Verify [2024] in answer text is NOT treated as a citation."""
    akus = _make_akus(3)
    answer = "In [2024], this happened according to [1] and [2]."

    updated, legend = resolve_aku_legend(answer, akus, {})

    # [2024] should be untouched, [1] and [2] re-indexed
    assert "[2024]" in updated
    assert "[1]" in updated
    assert "[2]" in updated


def test_resolve_aku_legend_handles_grouped_citations():
    """Verify [1, 2] grouped citations work correctly."""
    akus = _make_akus(3)
    answer = "Based on [1, 2] and [3]."

    updated, legend = resolve_aku_legend(answer, akus, {})

    assert "[1, 2]" in updated
    assert "[3]" in updated
    assert "Sources" in legend


def test_resolve_aku_legend_no_citations():
    """Verify no legend is generated when there are no citations."""
    akus = _make_akus(3)
    answer = "This answer has no citations."

    updated, legend = resolve_aku_legend(answer, akus, {})

    assert updated == answer
    assert legend == ""


def test_check_faithfulness_ignores_years():
    """Verify [2024] is not counted as a citation in faithfulness check."""
    akus = _make_akus(3)
    answer = "In [2024], something happened [1] and [2] and [3]."

    result = check_faithfulness(answer, akus)

    # Should not report "2024" as hallucinated
    assert result["is_faithful"] is True


def test_check_faithfulness_detects_hallucinated_index():
    """Verify hallucinated citation indices are caught."""
    akus = _make_akus(2)
    answer = "According to [1] and [5], this is true."

    result = check_faithfulness(answer, akus)

    assert result["is_faithful"] is False
    assert "5" in str(result["issue"])
