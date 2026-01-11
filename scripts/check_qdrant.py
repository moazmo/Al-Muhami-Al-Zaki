#!/usr/bin/env python3
"""Check what's currently in Qdrant vector database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from src.utils.config import get_settings

settings = get_settings()
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

# Get collection info
info = client.get_collection("egyptian_law")
print(f"Total vectors: {info.points_count}")

# Sample documents
results = client.scroll(
    collection_name="egyptian_law", limit=200, with_payload=True, with_vectors=False
)

sources = {}
for point in results[0]:
    src = point.payload.get("source_name", "Unknown")
    if src not in sources:
        sources[src] = 0
    sources[src] += 1

print("\n=== Documents by Source ===")
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count} chunks")
