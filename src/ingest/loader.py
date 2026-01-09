"""
Document loader for Al-Muhami Al-Zaki.

Handles loading legal documents from various formats:
- PDF (via pypdf/pdfplumber)
- TXT
- DOCX
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

# Lazy imports to avoid loading heavy libraries at module import
_PDF_AVAILABLE = False
_DOCX_AVAILABLE = False


def _ensure_pdf():
    """Lazy load PDF libraries."""
    global _PDF_AVAILABLE
    try:
        import pypdf
        import pdfplumber

        _PDF_AVAILABLE = True
    except ImportError:
        raise ImportError("PDF support requires: pip install pypdf pdfplumber")


def _ensure_docx():
    """Lazy load DOCX library."""
    global _DOCX_AVAILABLE
    try:
        import docx

        _DOCX_AVAILABLE = True
    except ImportError:
        raise ImportError("DOCX support requires: pip install python-docx")


class DocumentLoader:
    """
    Loader for legal documents (PDF, TXT, DOCX).

    Extracts raw text while preserving structure.
    Does NOT perform anonymization (see anonymizer.py).

    Example:
        loader = DocumentLoader()
        text, metadata = loader.load("data/raw/civil_code.pdf")
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize the document loader.

        Args:
            use_pdfplumber: If True, use pdfplumber for PDFs (better for tables).
                           If False, use pypdf (faster but less accurate).
        """
        self.use_pdfplumber = use_pdfplumber

    def load(
        self, file_path: Union[str, Path], metadata: Optional[Dict] = None
    ) -> tuple[str, Dict]:
        """
        Load a document and extract text.

        Args:
            file_path: Path to the document
            metadata: Optional base metadata to include

        Returns:
            Tuple of (extracted_text, metadata_dict)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Loading document: {path.name}")

        # Extract text based on file type
        if suffix == ".pdf":
            text = self._load_pdf(path)
        elif suffix == ".txt":
            text = self._load_txt(path)
        elif suffix == ".docx":
            text = self._load_docx(path)
        else:
            raise ValueError(f"Unhandled file type: {suffix}")

        # Build metadata
        base_metadata = {
            "source_file": str(path.absolute()),
            "file_name": path.name,
            "file_type": suffix,
            "char_count": len(text),
        }

        if metadata:
            base_metadata.update(metadata)

        logger.info(f"Extracted {len(text):,} characters from {path.name}")

        return text, base_metadata

    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        _ensure_pdf()

        if self.use_pdfplumber:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n\n".join(text_parts)
        else:
            import pypdf

            text_parts = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n\n".join(text_parts)

    def _load_txt(self, path: Path) -> str:
        """Load text file with UTF-8 encoding."""
        return path.read_text(encoding="utf-8")

    def _load_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        _ensure_docx()

        import docx

        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        return "\n\n".join(paragraphs)

    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        metadata: Optional[Dict] = None,
    ) -> List[tuple[str, Dict]]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to directory
            recursive: If True, search subdirectories
            metadata: Base metadata to apply to all documents

        Returns:
            List of (text, metadata) tuples
        """
        dir_path = Path(directory)

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        pattern = "**/*" if recursive else "*"

        results = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                try:
                    text, file_metadata = self.load(file_path, metadata)
                    results.append((text, file_metadata))
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(results)} documents from {dir_path}")
        return results
