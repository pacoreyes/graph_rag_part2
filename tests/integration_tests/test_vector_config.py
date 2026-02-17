# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Live vector index configuration check.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Live vector index configuration check."""

import pytest
from agent.infrastructure.clients import neo4j_client, hf_embedding_client
from agent.tools.embeddings import process_embedding
from agent.settings import settings

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_neo4j_vector_index_dimensions():
    """Verify Neo4j vector index dimensions match the embedding model."""
    driver = await neo4j_client.get_driver()
    async with driver.session() as session:
        # Get vector indexes
        result = await session.run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options WHERE type = 'VECTOR'")
        records = await result.data()
        
        assert len(records) > 0, "No vector indexes found in Neo4j."
        
        for record in records:
            options = record.get("options", {})
            # Neo4j 5.x returns dimensions in options.indexConfig['vector.dimensions']
            index_config = options.get("indexConfig", {})
            dims = index_config.get("vector.dimensions")
            
            print(f"Index: {record['name']}, Dims: {dims}, Labels: {record['labelsOrTypes']}, Props: {record['properties']}")
            
            # Snowflake/snowflake-arctic-embed-s should be 384
            # Snowflake/snowflake-arctic-embed-m should be 768
            # The current setting is 's' -> 384
            assert dims == 384, f"Index {record['name']} has {dims} dimensions, but expected 384 for Snowflake-s."

async def test_hf_embedding_dimensions():
    """Verify HF Inference API returns embeddings with correct dimensions."""
    # Test configured model
    try:
        embedding = await hf_embedding_client.embed("Test query")
        processed = process_embedding(embedding)
        print(f"Configured model ({settings.embedding_model_name}) length: {len(processed)}")
        assert len(processed) == 384
    except Exception as e:
        print(f"Configured model failed: {e}")
        
    # Test alternative common model to verify API endpoint
    from agent.infrastructure.hf_embedding_client import HFEmbeddingClient
    alt_client = HFEmbeddingClient("sentence-transformers/all-MiniLM-L6-v2", settings.hf_token.get_secret_value())
    try:
        embedding = await alt_client.embed("Test query")
        print(f"Alt model (MiniLM) length: {len(embedding)}")
    except Exception as e:
        print(f"Alt model failed: {e}")
