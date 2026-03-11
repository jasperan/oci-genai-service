"""Document loaders — Docling for rich formats, plain text fallback."""

from __future__ import annotations

from pathlib import Path

_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".xlsx"}


def load_document(file_path: str) -> str:
    """Load a document and return its text content.

    Uses Docling for PDF/DOCX/PPTX/HTML, plain text for .txt/.md.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in _DOCLING_EXTENSIONS:
        return _load_with_docling(file_path)
    else:
        return path.read_text(encoding="utf-8")


def _load_with_docling(file_path: str) -> str:
    """Load a document using Docling for rich structural parsing."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()
