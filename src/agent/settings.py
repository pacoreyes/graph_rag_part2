"""Application settings loaded from environment variables."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for the application.

    Loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: SecretStr

    # Gemini
    gemini_api_key: SecretStr

    # Pinecone
    pinecone_api_key: SecretStr
    pinecone_index_chunks_name: str = "chunks"
    pinecone_index_community_summaries: str = "community-summaries"

    # Nomic embedding model
    nomic_model_name: str = "nomic-ai/nomic-embed-text-v1.5"

    # Data volume path
    data_volume_path: str = "data_volume/assets"


# Initialize singleton settings
settings = Settings()
