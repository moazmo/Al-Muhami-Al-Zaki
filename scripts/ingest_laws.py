"""
Legal Document Ingestion Script for Al-Muhami Al-Zaki.

Usage:
    python scripts/ingest_laws.py --input data/raw/civil_code.pdf
    python scripts/ingest_laws.py --input data/raw/ --recursive
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.ingest.loader import DocumentLoader
from src.ingest.anonymizer import ArabicAnonymizer
from src.ingest.chunker import LegalChunker
from src.ingest.embedder import LegalEmbedder
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into Al-Muhami Al-Zaki"
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to PDF/TXT file or directory"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If input is directory, search recursively",
    )

    parser.add_argument(
        "--source-name", type=str, default="قانون مصري", help="Name of the legal source"
    )

    parser.add_argument(
        "--source-type",
        type=str,
        choices=["law", "ruling", "regulation", "constitution"],
        default="law",
        help="Type of legal document",
    )

    parser.add_argument(
        "--law-number", type=str, default=None, help="Law number (e.g., '131')"
    )

    parser.add_argument("--law-year", type=int, required=True, help="Year of enactment")

    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Maximum chunk size in characters"
    )

    parser.add_argument(
        "--skip-anonymization",
        action="store_true",
        help="Skip PII anonymization (use for laws, not rulings)",
    )

    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate Qdrant collection",
    )

    return parser.parse_args()


def main():
    """Main ingestion pipeline."""
    args = parse_args()

    # Setup logging
    setup_logger(level="INFO")

    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — Document Ingestion Pipeline")
    logger.info("=" * 60)

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Base metadata for all chunks
    base_metadata = {
        "source_name": args.source_name,
        "source_type": args.source_type,
        "law_number": args.law_number,
        "law_year": args.law_year,
    }

    # -------------------------------------------------------------------------
    # Step 1: Load Documents
    # -------------------------------------------------------------------------
    logger.info("Step 1: Loading documents...")
    loader = DocumentLoader(use_pdfplumber=False)

    if input_path.is_file():
        text, metadata = loader.load(input_path, base_metadata)
        documents = [(text, {**base_metadata, **metadata})]
    else:
        documents = loader.load_directory(
            input_path, recursive=args.recursive, metadata=base_metadata
        )

    if not documents:
        logger.error("No documents loaded!")
        return 1

    logger.info(f"Loaded {len(documents)} document(s)")

    # -------------------------------------------------------------------------
    # Step 2: Anonymize (Optional)
    # -------------------------------------------------------------------------
    if not args.skip_anonymization:
        logger.info("Step 2: Anonymizing PII (Law 151 compliance)...")

        try:
            anonymizer = ArabicAnonymizer()
        except Exception as e:
            logger.warning(f"NER model unavailable, using regex fallback: {e}")
            from src.ingest.anonymizer import SimpleAnonymizer

            anonymizer = SimpleAnonymizer()

        anonymized_documents = []
        total_entities = 0

        for text, metadata in documents:
            anonymized_text, audit_log = anonymizer.anonymize(text)
            total_entities += len(audit_log)

            doc_data = {
                **metadata,
                "text": text,  # Original for display
                "text_anonymized": anonymized_text,  # For embedding
                "is_anonymized": True,
            }
            anonymized_documents.append(doc_data)

        logger.info(f"Anonymized {total_entities} entities")
        documents = anonymized_documents
    else:
        logger.info("Step 2: Skipping anonymization (--skip-anonymization)")

        # Still structure the data properly
        documents = [
            {**metadata, "text": text, "text_anonymized": text, "is_anonymized": False}
            for text, metadata in documents
        ]

    # -------------------------------------------------------------------------
    # Step 3: Chunk Documents
    # -------------------------------------------------------------------------
    logger.info("Step 3: Chunking documents...")
    chunker = LegalChunker(chunk_size=args.chunk_size)

    all_chunks = []
    for doc in documents:
        text = doc.get("text_anonymized", doc.get("text", ""))
        chunks = chunker.chunk(text, doc)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks")

    # -------------------------------------------------------------------------
    # Step 4: Embed and Upload
    # -------------------------------------------------------------------------
    logger.info("Step 4: Embedding and uploading to Qdrant...")
    embedder = LegalEmbedder()

    # Create collection if needed
    embedder.create_collection(recreate=args.recreate_collection)

    # Upload chunks
    uploaded = embedder.embed_and_upload(all_chunks)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Ingestion Complete!")
    logger.info(f"  Documents processed: {len(documents)}")
    logger.info(f"  Chunks created: {len(all_chunks)}")
    logger.info(f"  Vectors uploaded: {uploaded}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
