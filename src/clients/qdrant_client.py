"""
Qdrant client wrapper for Al-Muhami Al-Zaki.

Provides configured Qdrant client for vector operations.
"""

from typing import Optional

from qdrant_client import QdrantClient
from loguru import logger

from src.utils.config import get_settings


_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Get a configured Qdrant client.

    Uses singleton pattern to reuse client across calls.

    Returns:
        Configured QdrantClient
    """
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    settings = get_settings()

    logger.info(f"Initializing Qdrant client: {settings.qdrant_url}")

    _qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )

    return _qdrant_client


def get_collection_info() -> dict:
    """
    Get information about the legal documents collection.

    Returns:
        Collection info dict with count, vectors_count, etc.
    """
    client = get_qdrant_client()
    settings = get_settings()

    try:
        info = client.get_collection(settings.qdrant_collection_name)
        return {
            "name": settings.qdrant_collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return {"error": str(e)}
