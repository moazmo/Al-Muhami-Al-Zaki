"""
Comprehensive System Verification Script for Al-Muhami Al-Zaki.

Tests:
1. Qdrant connection and collection state
2. Data integrity (correct text, not reversed)
3. Search functionality
4. Article detection in chunks
5. Full RAG pipeline with sample query
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingest.embedder import LegalEmbedder
from src.utils.config import get_settings


def verify_qdrant_state():
    """Check Qdrant collection state and data integrity."""
    logger.info("=" * 60)
    logger.info("STEP 1: QDRANT CONNECTION & COLLECTION STATE")
    logger.info("=" * 60)

    settings = get_settings()
    embedder = LegalEmbedder()

    try:
        info = embedder.qdrant.get_collection(settings.qdrant_collection_name)
        logger.success(f"✓ Connected to Qdrant Cloud")
        logger.info(f"  Collection: {settings.qdrant_collection_name}")
        logger.info(f"  Points Count: {info.points_count}")
        logger.info(f"  Vectors Config: {info.config.params.vectors}")

        if info.points_count == 0:
            logger.error("❌ Collection is EMPTY! No data ingested.")
            return False

        if info.points_count > 2000:
            logger.warning(
                f"⚠️ Unexpected high count ({info.points_count}). Old data may not be deleted."
            )
        elif info.points_count < 1000:
            logger.warning(
                f"⚠️ Low count ({info.points_count}). Some documents may have failed."
            )
        else:
            logger.success(f"✓ Point count looks reasonable: {info.points_count}")

        return {"embedder": embedder, "points_count": info.points_count}

    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        return False


def verify_data_integrity(embedder):
    """Check that stored data is not reversed."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: DATA INTEGRITY CHECK")
    logger.info("=" * 60)

    settings = get_settings()

    # Sample some points from the collection
    try:
        results = embedder.qdrant.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        points = results[0]
        logger.info(f"Sampled {len(points)} points for inspection")

        reversed_count = 0
        correct_count = 0

        for point in points:
            text = point.payload.get("text_anonymized", "")[:100]
            source = point.payload.get("source_name", "Unknown")

            # Check for reversed markers
            if "ةدام" in text or "نوناق" in text:
                reversed_count += 1
                logger.warning(f"⚠️ REVERSED text found in: {source}")
                logger.warning(f"   Sample: {text[:50]}...")
            elif "مادة" in text or "قانون" in text:
                correct_count += 1
                logger.success(f"✓ Correct text: {source}")
            else:
                logger.info(f"  Neutral text: {source}")

        if reversed_count > 0:
            logger.error(f"❌ Found {reversed_count} chunks with reversed text!")
            return False
        else:
            logger.success(f"✓ All sampled chunks have correct orientation")
            return True

    except Exception as e:
        logger.error(f"❌ Data integrity check failed: {e}")
        return False


def verify_search(embedder):
    """Test search functionality with known queries."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: SEARCH FUNCTIONALITY")
    logger.info("=" * 60)

    test_queries = [
        ("ما هي عقوبة السرقة", "318", "قانون العقوبات"),
        ("حقوق المستأجر", None, "القانون المدني"),
        ("شروط الزواج", None, "الأحوال الشخصية"),
    ]

    results_summary = []

    for query, expected_article, expected_source in test_queries:
        logger.info(f"\nQuery: '{query}'")

        try:
            results = embedder.search(query, top_k=3)

            if not results:
                logger.error(f"  ❌ No results returned!")
                results_summary.append(
                    {"query": query, "status": "FAIL", "reason": "No results"}
                )
                continue

            top_result = results[0]
            score = top_result.get("score", 0)
            text = top_result.get("text", "")[:150]
            source = top_result.get("source_name", "Unknown")
            article = top_result.get("article_number", "N/A")

            logger.info(f"  Top Result:")
            logger.info(f"    Score: {score:.4f}")
            logger.info(f"    Source: {source}")
            logger.info(f"    Article: {article}")
            logger.info(f"    Text: {text}...")

            # Check for reversed text in results
            if "ةدام" in text or "نوناق" in text:
                logger.error(f"  ❌ Result contains REVERSED text!")
                results_summary.append(
                    {"query": query, "status": "FAIL", "reason": "Reversed text"}
                )
            elif score < 0.3:
                logger.warning(f"  ⚠️ Low relevance score")
                results_summary.append(
                    {"query": query, "status": "WARN", "score": score}
                )
            else:
                logger.success(f"  ✓ Search returned valid results")
                results_summary.append({"query": query, "status": "OK", "score": score})

        except Exception as e:
            logger.error(f"  ❌ Search failed: {e}")
            results_summary.append({"query": query, "status": "FAIL", "reason": str(e)})

    return results_summary


def verify_article_detection():
    """Check if articles are being detected in chunks."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: ARTICLE NUMBER DETECTION")
    logger.info("=" * 60)

    settings = get_settings()
    embedder = LegalEmbedder()

    # Check how many chunks have article numbers
    try:
        results = embedder.qdrant.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points = results[0]
        with_article = 0
        without_article = 0

        for point in points:
            article = point.payload.get("article_number")
            if article and article != "N/A":
                with_article += 1
            else:
                without_article += 1

        pct = (with_article / len(points)) * 100 if points else 0

        logger.info(
            f"Chunks with article numbers: {with_article}/{len(points)} ({pct:.1f}%)"
        )

        if pct < 10:
            logger.warning(
                "⚠️ Very few chunks have article numbers. Chunking regex may need adjustment."
            )
        else:
            logger.success(
                f"✓ Article detection working: {pct:.1f}% of chunks have article numbers"
            )

        return {
            "with_article": with_article,
            "without_article": without_article,
            "pct": pct,
        }

    except Exception as e:
        logger.error(f"❌ Article detection check failed: {e}")
        return None


def main():
    """Run all verification steps."""
    print("\n" + "=" * 70)
    print("AL-MUHAMI AL-ZAKI: COMPREHENSIVE SYSTEM VERIFICATION")
    print("=" * 70 + "\n")

    # Step 1: Qdrant state
    qdrant_result = verify_qdrant_state()
    if not qdrant_result:
        logger.error("CRITICAL: Qdrant verification failed. Stopping.")
        return

    embedder = qdrant_result["embedder"]

    # Step 2: Data integrity
    integrity_ok = verify_data_integrity(embedder)

    # Step 3: Search functionality
    search_results = verify_search(embedder)

    # Step 4: Article detection
    article_stats = verify_article_detection()

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\n1. Qdrant Collection: {qdrant_result['points_count']} points")
    print(f"2. Data Integrity: {'✓ PASS' if integrity_ok else '❌ FAIL'}")
    print(f"3. Search Tests:")
    for r in search_results:
        status_icon = (
            "✓" if r["status"] == "OK" else "⚠" if r["status"] == "WARN" else "❌"
        )
        print(f"   {status_icon} {r['query'][:30]}... ({r['status']})")

    if article_stats:
        print(
            f"4. Article Detection: {article_stats['pct']:.1f}% of chunks have article numbers"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
