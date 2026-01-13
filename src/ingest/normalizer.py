"""
Arabic Text Normalizer for Al-Muhami Al-Zaki.

Fixes reversed Arabic text extracted from PDFs where the text layer
is stored backwards (common in older Arabic PDFs).

Example:
    Input:  'ىرصملا تابوقعلا نوناق'  (backwards)
    Output: 'قانون العقوبات المصري'  (correct)
"""

import re
from typing import Tuple

from loguru import logger


# Markers to detect reversed text
REVERSED_MARKERS = [
    "ةدام",  # Reversed "مادة" (Article)
    "نوناق",  # Reversed "قانون" (Law)
    "ةداملا",  # Reversed "المادة" (The Article)
]

CORRECT_MARKERS = [
    "مادة",  # Article
    "قانون",  # Law
    "المادة",  # The Article
]


def is_text_reversed(text: str) -> bool:
    """
    Detect if Arabic text is reversed by checking for known markers.

    Args:
        text: Text to check

    Returns:
        True if text appears to be reversed
    """
    # Count occurrences of reversed vs correct markers
    reversed_count = sum(text.count(marker) for marker in REVERSED_MARKERS)
    correct_count = sum(text.count(marker) for marker in CORRECT_MARKERS)

    # If we find more reversed markers than correct ones, text is reversed
    if reversed_count > correct_count and reversed_count > 5:
        return True

    # Also check if common legal terms are backwards
    # "العقوبات" correct vs "تابوقعلا" reversed
    if text.count("تابوقعلا") > text.count("العقوبات"):
        return True

    return False


def reverse_arabic_line(line: str) -> str:
    """
    Reverse a single line of Arabic text while preserving:
    - Numbers (Arabic numerals stay in place)
    - Punctuation at line boundaries
    - Whitespace structure

    Args:
        line: A single line of text

    Returns:
        Reversed line
    """
    if not line.strip():
        return line

    # Split line into segments: numbers vs text
    # Pattern matches: Arabic text, numbers, or other
    segments = re.findall(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+|[\d\u0660-\u0669]+|[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\d\u0660-\u0669]+",
        line,
    )

    result_segments = []
    for segment in segments:
        # Check if segment is Arabic text
        if re.match(
            r"^[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+$", segment
        ):
            # Reverse Arabic text
            result_segments.append(segment[::-1])
        else:
            # Keep numbers and punctuation as-is
            result_segments.append(segment)

    # Reverse the order of segments (RTL -> LTR visual order fix)
    result_segments.reverse()

    return "".join(result_segments)


def normalize_reversed_text(text: str) -> Tuple[str, bool]:
    """
    Normalize reversed Arabic text from PDF extraction.

    Args:
        text: Raw extracted text (potentially reversed)

    Returns:
        Tuple of (normalized_text, was_reversed)
    """
    if not text:
        return text, False

    # Check if text needs reversal
    if not is_text_reversed(text):
        logger.info("Text appears correctly oriented, no reversal needed")
        return text, False

    logger.warning("Detected REVERSED Arabic text, applying correction...")

    # Process line by line
    lines = text.split("\n")
    corrected_lines = []

    for line in lines:
        corrected_lines.append(reverse_arabic_line(line))

    result = "\n".join(corrected_lines)

    # Verify correction worked
    if is_text_reversed(result):
        logger.error("Text still appears reversed after correction!")
    else:
        logger.success("Text successfully corrected!")

    return result, True


def normalize_pdf_text(text: str, filename: str = "") -> str:
    """
    Main entry point for normalizing PDF text.

    Automatically detects and fixes reversed Arabic text.

    Args:
        text: Raw text from PDF extraction
        filename: Optional filename for logging

    Returns:
        Normalized text ready for chunking
    """
    if not text:
        return text

    prefix = f"[{filename}] " if filename else ""

    # Check and fix reversal
    normalized, was_reversed = normalize_reversed_text(text)

    if was_reversed:
        logger.info(f"{prefix}Applied text reversal correction")

        # Log a sample to verify
        sample = normalized[:200].replace("\n", " ")
        logger.debug(f"{prefix}Sample after fix: {sample}...")

    return normalized


if __name__ == "__main__":
    # Test with sample reversed text
    test_reversed = "ىرصملا تابوقعلا نوناق"
    test_correct = "قانون العقوبات المصري"

    print(f"Testing reversal detection...")
    print(f"  Reversed text detected: {is_text_reversed(test_reversed)}")
    print(f"  Correct text detected as reversed: {is_text_reversed(test_correct)}")

    print(f"\nTesting line reversal...")
    result = reverse_arabic_line(test_reversed)
    print(f"  Input:  {test_reversed}")
    print(f"  Output: {result}")
