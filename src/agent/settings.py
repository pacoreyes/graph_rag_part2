# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Application settings loaded from environment variables.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Application settings loaded from environment variables."""

from pydantic import Field, SecretStr
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

    # Hugging Face (for Nomic embeddings)
    hf_token: SecretStr = Field(alias="HUGGING_FACE_HUB_TOKEN")

    # Pinecone
    pinecone_api_key: SecretStr
    pinecone_index_chunks_name: str = "chunks"
    pinecone_index_community_summaries: str = "community-summaries"

    # Primary embedding model (HF Inference API)
    embedding_model_name: str = "Snowflake/snowflake-arctic-embed-s"

    # Data volume path
    data_volume_path: str = "data_volume/assets"


# Initialize singleton settings
settings = Settings()
