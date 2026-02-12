import tempfile
from pathlib import Path

import polars as pl

from agent.tools.source_resolver import resolve_source_urls


def _create_test_parquet(tmp_dir: str) -> str:
    """Create a temp parquet file with test data."""
    path = str(Path(tmp_dir) / "wikipedia_articles.parquet")
    df = pl.DataFrame(
        {
            "id": ["Q1", "Q2", "Q3"],
            "name": ["Alice", "Bob", "Charlie"],
            "wikipedia_url": [
                "https://en.wikipedia.org/wiki/Alice",
                "https://en.wikipedia.org/wiki/Bob",
                "https://en.wikipedia.org/wiki/Charlie",
            ],
        }
    )
    df.write_parquet(path)
    return path


def test_resolve_source_urls_happy_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _create_test_parquet(tmp_dir)

        result = resolve_source_urls(["Q1", "Q3"], path)

    assert len(result) == 2
    assert result["Q1"]["name"] == "Alice"
    assert result["Q3"]["wikipedia_url"] == "https://en.wikipedia.org/wiki/Charlie"


def test_resolve_source_urls_empty_qids():
    result = resolve_source_urls([], "/nonexistent.parquet")

    assert result == {}


def test_resolve_source_urls_unknown_qids():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _create_test_parquet(tmp_dir)

        result = resolve_source_urls(["Q999", "Q888"], path)

    assert result == {}


def test_resolve_source_urls_deduplicates_qids():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _create_test_parquet(tmp_dir)

        result = resolve_source_urls(["Q1", "Q1", "Q1"], path)

    assert len(result) == 1
    assert result["Q1"]["name"] == "Alice"
