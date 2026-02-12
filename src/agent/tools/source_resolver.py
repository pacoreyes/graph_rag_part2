"""Resolve Wikidata QIDs to source URLs via parquet lookup.

Pure function with dependency injection — no global config or singletons.
"""

import polars as pl


def resolve_source_urls(
    qids: list[str],
    parquet_path: str,
) -> dict[str, dict[str, str]]:
    """Resolve a list of Wikidata QIDs to name and Wikipedia URL.

    Uses Polars lazy scan for memory-efficient parquet reading.

    Args:
        qids: List of Wikidata QID strings to resolve.
        parquet_path: Path to the wikipedia_articles.parquet file.

    Returns:
        dict[str, dict[str, str]]: Mapping of QID to {"name": ..., "wikipedia_url": ...}.
    """
    if not qids:
        return {}

    unique_qids = list(set(qids))
    lf = pl.scan_parquet(parquet_path)
    result = (
        lf.filter(pl.col("id").is_in(unique_qids))
        .select("id", "name", "wikipedia_url")
        .collect()
    )

    return {
        row["id"]: {"name": row["name"], "wikipedia_url": row["wikipedia_url"]}
        for row in result.iter_rows(named=True)
    }
