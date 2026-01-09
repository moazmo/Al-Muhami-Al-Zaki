"""
Quick retrieval test script for Al-Muhami Al-Zaki.

Usage:
    python scripts/test_retrieval.py --query "ما هي عقوبة السرقة؟"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.ingest.embedder import LegalEmbedder
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test retrieval from Al-Muhami Al-Zaki"
    )

    parser.add_argument(
        "--query", type=str, required=True, help="Legal question to search for"
    )

    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )

    parser.add_argument(
        "--source-type",
        type=str,
        choices=["law", "ruling", "regulation", "constitution"],
        default=None,
        help="Filter by document type",
    )

    return parser.parse_args()


def main():
    """Run retrieval test."""
    args = parse_args()

    setup_logger(level="INFO")

    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — Retrieval Test")
    logger.info("=" * 60)

    logger.info(f"Query: {args.query}")

    # Build filters
    filters = {}
    if args.source_type:
        filters["source_type"] = args.source_type

    # Run search
    embedder = LegalEmbedder()
    results = embedder.search(
        query=args.query,
        top_k=args.top_k,
        filters=filters if filters else None,
    )

    # Display results
    logger.info(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{'=' * 60}")
        print(f"Result {i} (Score: {result.get('score', 0):.4f})")
        print(f"{'=' * 60}")
        print(f"Source: {result.get('source_name', 'Unknown')}")
        print(f"Article: {result.get('article_number', 'N/A')}")
        print(f"Year: {result.get('law_year', 'N/A')}")
        print(f"Type: {result.get('source_type', 'N/A')}")
        print(f"\nText:\n{result.get('text', '')[:500]}...")
        print()

    return 0


if __name__ == "__main__":
    exit(main())
