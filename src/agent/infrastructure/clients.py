# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Client wiring layer.
# This is the only module (besides settings.py) that reads secrets
# and instantiates infrastructure clients. All other modules receive
# injected dependencies.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------


from agent.infrastructure.gemini_client import GeminiClient
from agent.infrastructure.neo4j_client import Neo4jClient
from agent.infrastructure.hf_embedding_client import HFEmbeddingClient
from agent.infrastructure.pinecone_client import PineconeClient
from agent.settings import settings

neo4j_client = Neo4jClient(
    uri=settings.neo4j_uri,
    username=settings.neo4j_username,
    password=settings.neo4j_password.get_secret_value(),
)

pinecone_client = PineconeClient(
    api_key=settings.pinecone_api_key.get_secret_value(),
)

gemini_client = GeminiClient(
    api_key=settings.gemini_api_key.get_secret_value(),
    schema_path=f"{settings.data_volume_path}/graph_schema.json",
)

hf_embedding_client = HFEmbeddingClient(
    model_name=settings.embedding_model_name,
    api_token=settings.hf_token.get_secret_value(),
)
