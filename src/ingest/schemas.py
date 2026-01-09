"""
Pydantic schemas for legal document chunks.

Defines the payload structure stored in Qdrant.
Compliant with Egyptian Data Protection Law 151/2020.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class LegalChunkPayload(BaseModel):
    """
    Qdrant payload schema for legal document chunks.

    Every chunk stored in Qdrant MUST conform to this schema.
    The schema enforces Law 151/2020 compliance through required
    anonymization tracking fields.

    Attributes:
        text: Original text for display to user after retrieval
        text_anonymized: Anonymized text used for embedding (Law 151 compliance)
        source_name: Human-readable document title in Arabic
        source_type: Document category (law, ruling, regulation, constitution)
        law_number: Law identifier (null for constitution)
        law_year: Year of enactment (critical for temporal filtering)
        article_number: Article identifier (e.g., "157", "104-مكرر")
        chapter: Chapter/Section title for hierarchical navigation
        chunk_index: Position within multi-chunk articles
        total_chunks: Total chunks for this article (for reassembly)
        is_anonymized: Flag indicating PII removal status
        ingestion_timestamp: ISO 8601 timestamp for data lineage
    """

    # Core content
    text: str = Field(
        ..., min_length=1, description="Original text for display to user"
    )
    text_anonymized: str = Field(
        ...,
        min_length=1,
        description="Anonymized text for embedding (Law 151 compliant)",
    )

    # Source identification
    source_name: str = Field(
        ...,
        min_length=1,
        description="Document title in Arabic (e.g., 'القانون المدني المصري')",
    )
    source_type: Literal["law", "ruling", "regulation", "constitution"] = Field(
        ..., description="Document category for filtering"
    )

    # Legal reference fields
    law_number: Optional[str] = Field(
        None, description="Law number (e.g., '131' for Civil Code)"
    )
    law_year: int = Field(..., ge=1800, le=2100, description="Year of enactment")
    article_number: Optional[str] = Field(
        None, description="Article identifier (e.g., '157', '104-مكرر')"
    )
    chapter: Optional[str] = Field(None, description="Chapter/Section title")

    # Chunk metadata
    chunk_index: int = Field(
        default=0, ge=0, description="Position of this chunk within the article"
    )
    total_chunks: int = Field(
        default=1, ge=1, description="Total number of chunks for this article"
    )

    # Compliance tracking
    is_anonymized: bool = Field(
        default=True, description="Flag indicating PII has been removed"
    )
    ingestion_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this chunk was ingested"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "يلزم المتعاقد بالوفاء بما تعهد به...",
                "text_anonymized": "يلزم المتعاقد بالوفاء بما تعهد به...",
                "source_name": "القانون المدني المصري",
                "source_type": "law",
                "law_number": "131",
                "law_year": 1948,
                "article_number": "147",
                "chapter": "الباب الأول - الالتزامات",
                "chunk_index": 0,
                "total_chunks": 1,
                "is_anonymized": True,
                "ingestion_timestamp": "2026-01-03T20:00:00Z",
            }
        }


class AnonymizationAuditLog(BaseModel):
    """
    Audit log entry for anonymization operations.

    Used to track what entities were redacted for Law 151 compliance.
    """

    entity_type: str = Field(..., description="Type of entity (PER, LOC, ORG)")
    original_text: str = Field(..., description="Original text that was redacted")
    replacement: str = Field(..., description="Replacement mask used")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="NER model confidence score"
    )
    start_position: int = Field(
        ..., ge=0, description="Character start position in original text"
    )
    end_position: int = Field(
        ..., ge=0, description="Character end position in original text"
    )


class IngestionResult(BaseModel):
    """
    Result of an ingestion operation.
    """

    source_file: str = Field(..., description="Path to source file")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    entities_anonymized: int = Field(..., ge=0, description="Total entities masked")
    success: bool = Field(..., description="Whether ingestion succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
