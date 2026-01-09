"""
Legal-aware text chunker for Al-Muhami Al-Zaki.

Splits Arabic legal documents at semantic boundaries:
- Articles (مادة)
- Chapters (باب)
- Sections (فصل)
- Clauses (بند)

Never splits mid-article.
"""

import re
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# Arabic legal structure delimiters (ordered by priority)
LEGAL_DELIMITERS = [
    r"\n(?=(?:مادة|المادة)\s*\(?[\u0660-\u0669\d]+)",  # Article markers
    r"\n(?=(?:الباب|باب)\s)",  # Chapter markers
    r"\n(?=(?:الفصل|فصل)\s)",  # Section markers
    r"\n(?=(?:البند|بند)\s)",  # Clause markers
    "\n\n",  # Double newline
    "\n",  # Single newline
    "،",  # Arabic comma
    " ",  # Space
]

# Regex to extract article number from text
ARTICLE_NUMBER_PATTERN = re.compile(
    r"(?:مادة|المادة)\s*\(?([‎\u0660-\u0669\d]+(?:\s*مكرر(?:\s*[أ-ي])?)?)\)?", re.UNICODE
)


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent processing.

    Operations:
    1. Remove diacritics (tashkeel)
    2. Normalize alef variants
    3. Normalize teh marbuta → heh (for search)
    4. Remove tatweel (kashida)
    5. Normalize whitespace

    Args:
        text: Raw Arabic text

    Returns:
        Normalized text
    """
    # Remove diacritics (harakat)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize alef variants (إأآا → ا)
    text = re.sub(r"[إأآا]", "ا", text)

    # Normalize teh marbuta (ة → ه) - for search matching
    # Note: Keep original for display
    text = re.sub(r"ة", "ه", text)

    # Remove tatweel (ـ)
    text = re.sub(r"\u0640", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


class LegalChunker:
    """
    Arabic Legal Document Chunker.

    Strategy: Recursive splitting with legal-structure awareness.
    Respects article boundaries and preserves legal context.

    Example:
        chunker = LegalChunker(chunk_size=1000)
        chunks = chunker.chunk(
            text="المادة 157...",
            metadata={"source_name": "القانون المدني", "law_year": 1948}
        )
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        normalize: bool = False,
    ):
        """
        Initialize the legal chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks for context
            normalize: If True, normalize Arabic text (affects search matching)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize = normalize

        self.splitter = RecursiveCharacterTextSplitter(
            separators=LEGAL_DELIMITERS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True,
        )

        logger.info(
            f"LegalChunker initialized: size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split legal text into semantic chunks.

        Each chunk is enriched with:
        - Extracted article number (if present)
        - Chunk index and total count
        - All provided metadata

        Args:
            text: Cleaned legal text (ideally already anonymized)
            metadata: Base metadata (source_name, law_year, etc.)

        Returns:
            List of chunk dictionaries ready for embedding
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []

        # Optionally normalize
        if self.normalize:
            text = normalize_arabic(text)

        # Split into chunks
        raw_chunks = self.splitter.split_text(text)

        if not raw_chunks:
            logger.warning("No chunks produced from text")
            return []

        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Try to extract article number from chunk
            article_match = ARTICLE_NUMBER_PATTERN.search(chunk_text)
            article_num = (
                article_match.group(1)
                if article_match
                else metadata.get("article_number")
            )

            chunk_data = {
                **metadata,
                "text": chunk_text,
                "article_number": article_num,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
            }
            chunks.append(chunk_data)

        logger.info(f"Created {len(chunks)} chunks from text")

        return chunks

    def chunk_by_article(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split text strictly by article boundaries.

        Each article becomes one chunk (may exceed chunk_size).
        Use for legal texts where article integrity is critical.

        Args:
            text: Legal text
            metadata: Base metadata

        Returns:
            List of article chunks
        """
        # Split at article markers
        article_pattern = re.compile(
            r"((?:مادة|المادة)\s*\(?[\u0660-\u0669\d]+(?:\s*مكرر(?:\s*[أ-ي])?)?[):]?\s*)",
            re.UNICODE,
        )

        parts = article_pattern.split(text)

        chunks = []
        current_article = None
        current_text = ""

        for part in parts:
            if article_pattern.match(part):
                # Save previous article
                if current_text.strip():
                    chunk_data = {
                        **metadata,
                        "text": current_text.strip(),
                        "article_number": current_article,
                        "chunk_index": len(chunks),
                    }
                    chunks.append(chunk_data)

                # Start new article
                article_match = ARTICLE_NUMBER_PATTERN.search(part)
                current_article = article_match.group(1) if article_match else None
                current_text = part
            else:
                current_text += part

        # Don't forget last article
        if current_text.strip():
            chunk_data = {
                **metadata,
                "text": current_text.strip(),
                "article_number": current_article,
                "chunk_index": len(chunks),
            }
            chunks.append(chunk_data)

        # Update total_chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        logger.info(f"Split into {len(chunks)} articles")

        return chunks
