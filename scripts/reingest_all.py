"""
Bulk Re-ingestion Script for Al-Muhami Al-Zaki.

This script:
1. Clears the existing Qdrant collection
2. Re-ingests all PDFs with text normalization (fixes reversed Arabic)
3. Verifies the new data

Usage:
    python scripts/reingest_all.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.ingest.loader import DocumentLoader
from src.ingest.normalizer import normalize_pdf_text, is_text_reversed
from src.ingest.chunker import LegalChunker
from src.ingest.embedder import LegalEmbedder
from src.utils.config import get_settings


# Document metadata mapping
DOCUMENTS = [
    {
        "file": "data/raw/Egyptian_Panel_code.pdf",
        "source_name": "قانون العقوبات المصري",
        "source_type": "law",
        "law_year": 1937,
    },
    {
        "file": "data/raw/civil_code_131_1948.pdf",
        "source_name": "القانون المدني",
        "source_type": "law",
        "law_year": 1948,
    },
    {
        "file": "data/raw/Egyptian Constitution 2014.pdf",
        "source_name": "الدستور المصري",
        "source_type": "constitution",
        "law_year": 2014,
    },
    {
        "file": "data/raw/Criminal Procedure Code.pdf",
        "source_name": "قانون الإجراءات الجنائية",
        "source_type": "law",
        "law_year": 1950,
    },
    {
        "file": "data/raw/Personal-Status-Law.pdf",
        "source_name": "قانون الأحوال الشخصية",
        "source_type": "law",
        "law_year": 2000,
    },
    {
        "file": "data/raw/family-laws-egypt.pdf",
        "source_name": "قوانين الأسرة المصرية",
        "source_type": "law",
        "law_year": 2000,
    },
]


def reingest_all(clear_first: bool = True, dry_run: bool = False):
    """
    Re-ingest all documents with text normalization.

    Args:
        clear_first: If True, clear the collection before ingesting
        dry_run: If True, don't actually upload to Qdrant
    """
    settings = get_settings()
    loader = DocumentLoader()
    chunker = LegalChunker(chunk_size=1000, chunk_overlap=100)
    embedder = LegalEmbedder()

    logger.info("=" * 60)
    logger.info("AL-MUHAMI AL-ZAKI: BULK RE-INGESTION")
    logger.info("=" * 60)

    # Step 1: Clear existing collection
    if clear_first and not dry_run:
        logger.warning("Clearing existing Qdrant collection...")
        try:
            embedder.qdrant.delete_collection(settings.qdrant_collection_name)
            logger.success(f"Deleted collection: {settings.qdrant_collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection (may not exist): {e}")

        # Recreate collection
        embedder.create_collection()
        logger.success("Created fresh collection")

    # Step 2: Process each document
    total_chunks = 0
    results = []

    for doc_config in DOCUMENTS:
        file_path = doc_config["file"]
        source_name = doc_config["source_name"]

        logger.info("-" * 40)
        logger.info(f"Processing: {source_name}")
        logger.info(f"File: {file_path}")

        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            results.append({"file": file_path, "status": "NOT_FOUND", "chunks": 0})
            continue

        try:
            # Load raw text
            raw_text, metadata = loader.load(file_path)
            logger.info(f"Loaded {len(raw_text)} characters")

            # Check if reversed
            was_reversed = is_text_reversed(raw_text)
            if was_reversed:
                logger.warning("⚠️ REVERSED text detected!")

            # Normalize (fix reversal)
            normalized_text = normalize_pdf_text(raw_text, Path(file_path).name)

            # Chunk
            chunk_metadata = {
                "source_name": source_name,
                "source_type": doc_config["source_type"],
                "law_year": doc_config["law_year"],
                "file_name": Path(file_path).name,
            }

            chunks = chunker.chunk(normalized_text, chunk_metadata)
            logger.info(f"Created {len(chunks)} chunks")

            # Upload to Qdrant
            if not dry_run:
                embedder.embed_and_upload(chunks)
                logger.success(f"Uploaded {len(chunks)} chunks to Qdrant")
            else:
                logger.info(f"[DRY RUN] Would upload {len(chunks)} chunks")

            total_chunks += len(chunks)
            results.append(
                {
                    "file": file_path,
                    "source": source_name,
                    "status": "OK",
                    "was_reversed": was_reversed,
                    "chunks": len(chunks),
                }
            )

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            results.append(
                {"file": file_path, "status": "ERROR", "error": str(e), "chunks": 0}
            )

    # Step 3: Summary
    logger.info("=" * 60)
    logger.info("RE-INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total chunks: {total_chunks}")

    # Print results table
    print("\n📊 RESULTS:")
    print("-" * 70)
    print(f"{'Source':<30} {'Status':<12} {'Reversed':<10} {'Chunks':<10}")
    print("-" * 70)

    for r in results:
        source = r.get("source", r["file"])[:28]
        status = r["status"]
        reversed_flag = "✓" if r.get("was_reversed") else ""
        chunks = r["chunks"]
        print(f"{source:<30} {status:<12} {reversed_flag:<10} {chunks:<10}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {'':<12} {'':<10} {total_chunks:<10}")

    # Verify collection
    if not dry_run:
        try:
            info = embedder.qdrant.get_collection(settings.qdrant_collection_name)
            logger.success(f"Collection now has {info.points_count} points")
        except Exception as e:
            logger.error(f"Could not verify collection: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-ingest all documents")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload to Qdrant")
    parser.add_argument(
        "--keep-existing", action="store_true", help="Don't clear collection first"
    )
    args = parser.parse_args()

    reingest_all(
        clear_first=not args.keep_existing,
        dry_run=args.dry_run,
    )
