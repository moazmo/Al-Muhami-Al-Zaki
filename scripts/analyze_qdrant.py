#!/usr/bin/env python3
"""Get sample of what's in Qdrant to create matching benchmark."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from src.utils.config import get_settings

settings = get_settings()
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

# Get sample documents
results = client.scroll(
    collection_name="egyptian_law", limit=100, with_payload=True, with_vectors=False
)

# Analyze sources and articles
data = []
for point in results[0]:
    payload = point.payload
    data.append(
        {
            "source": payload.get("source_name", "Unknown"),
            "article": payload.get("article_number", "N/A"),
            "text_preview": payload.get("text", "")[:100],
        }
    )

# Group by source
sources = {}
for d in data:
    src = d["source"]
    if src not in sources:
        sources[src] = {"count": 0, "articles": [], "samples": []}
    sources[src]["count"] += 1
    if d["article"] != "N/A" and d["article"] not in sources[src]["articles"]:
        sources[src]["articles"].append(d["article"])
    if len(sources[src]["samples"]) < 3:
        sources[src]["samples"].append(d["text_preview"])

# Save to file
with open("qdrant_analysis.json", "w", encoding="utf-8") as f:
    json.dump(sources, f, ensure_ascii=False, indent=2)

print("Saved to qdrant_analysis.json")
print(f"Found {len(sources)} sources")
for src, info in sources.items():
    print(f"  {src}: {info['count']} chunks, articles: {info['articles'][:5]}")
