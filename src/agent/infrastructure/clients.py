"""Client wiring layer.

This is the only module (besides settings.py) that reads secrets
and instantiates infrastructure clients. All other modules receive
injected dependencies.
"""

from agent.infrastructure.gemini_client import GeminiClient
from agent.infrastructure.neo4j_client import Neo4jClient
from agent.infrastructure.nomic_client import NomicClient
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
)

nomic_client = NomicClient(
    model_name=settings.nomic_model_name,
)
