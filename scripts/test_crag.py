"""
End-to-end CRAG pipeline test for Al-Muhami Al-Zaki.

Tests the full flow: Retrieve → Grade → Generate.

Usage:
    python scripts/test_crag.py --query "ما هي حقوق الملكية في القانون المدني؟"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.graph.builder import build_crag_graph
from src.graph.state import create_initial_state
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the full CRAG pipeline")

    parser.add_argument(
        "--query", type=str, required=True, help="Legal question to ask in Arabic"
    )

    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    return parser.parse_args()


async def run_crag_test(query: str, verbose: bool = False):
    """
    Run the full CRAG pipeline.

    Args:
        query: Legal question in Arabic
        verbose: Show detailed output
    """
    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — CRAG Pipeline Test")
    logger.info("=" * 60)

    logger.info(f"Query: {query}")

    # Build graph
    logger.info("Building CRAG graph...")
    graph = build_crag_graph()

    # Create initial state
    initial_state = create_initial_state(query)

    # Run the graph
    logger.info("Running CRAG pipeline...")
    logger.info("-" * 40)

    try:
        final_state = await graph.ainvoke(initial_state)

        # Extract results
        generation = final_state.get("generation", "No answer generated")
        graded_docs = final_state.get("graded_documents", [])
        rewrite_count = final_state.get("rewrite_count", 0)
        query_history = final_state.get("query_history", [])

        # Display results
        logger.info("-" * 40)
        logger.info("CRAG PIPELINE COMPLETE")
        logger.info("-" * 40)

        print("\n" + "=" * 60)
        print("📝 ANSWER")
        print("=" * 60)
        print(generation)

        print("\n" + "=" * 60)
        print("📚 SOURCES USED")
        print("=" * 60)

        if graded_docs:
            for i, doc in enumerate(graded_docs, 1):
                meta = doc.metadata
                print(f"\n[{i}] {meta.get('source_name', 'Unknown')}")
                print(f"    المادة: {meta.get('article_number', 'N/A')}")
                print(f"    السنة: {meta.get('law_year', 'N/A')}")
                print(f"    Score: {meta.get('score', 0):.4f}")
                if verbose:
                    print(f"    Text: {doc.page_content[:200]}...")
        else:
            print("No relevant documents found.")

        print("\n" + "=" * 60)
        print("📊 STATS")
        print("=" * 60)
        print(f"Documents graded: {len(graded_docs)}")
        print(f"Query rewrites: {rewrite_count}")
        if query_history:
            print(f"Query history: {query_history}")

        return final_state

    except Exception as e:
        logger.error(f"CRAG pipeline failed: {e}")
        raise


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logger(level="DEBUG" if args.verbose else "INFO")

    # Run async test
    try:
        asyncio.run(run_crag_test(args.query, args.verbose))
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
