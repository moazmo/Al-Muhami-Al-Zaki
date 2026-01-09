"""
Configuration loader for Al-Muhami Al-Zaki.

Loads environment variables with validation using Pydantic.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All sensitive values are loaded from .env file.
    """

    # -------------------------------------------------------------------------
    # API Keys
    # -------------------------------------------------------------------------
    groq_api_key: str = Field(..., description="Groq API key for Llama-3")
    google_api_key: str = Field(..., description="Google API key for Gemini")

    # -------------------------------------------------------------------------
    # Qdrant Configuration
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")
    qdrant_collection_name: str = Field(
        default="egyptian_law", description="Qdrant collection name"
    )

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        description="HuggingFace embedding model ID",
    )
    grader_model: str = Field(
        default="llama-3.1-8b-instant", description="Groq model for relevance grading"
    )
    generator_model: str = Field(
        default="gemini-1.5-flash", description="Gemini model for answer generation"
    )

    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level")
    retrieval_top_k: int = Field(
        default=5, ge=1, le=20, description="Number of documents to retrieve"
    )
    grading_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to keep document",
    )
    max_rewrite_attempts: int = Field(
        default=2, ge=0, le=5, description="Maximum query rewrite attempts"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Validated settings object

    Raises:
        ValidationError: If required environment variables are missing
    """
    return Settings()
