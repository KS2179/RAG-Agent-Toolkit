"""
ingestion/parsers/pdf_parser.py
Parser for PDF documents using pypdf.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ingestion.parsers.base import BaseParser, ParsedDocument


class PDFParser(BaseParser):
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package is required for PDF parsing. Install it via 'pip install pypdf'."
            )

        reader = pypdf.PdfReader(str(path))
        page_texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            page_texts.append(t)

        full_text = "\n\n".join(page_texts).strip()
        page_count = len(reader.pages)

        return ParsedDocument(
            text=full_text,
            source=path.name,
            metadata={
                "format": "pdf",
                "page_count": page_count,
            },
        )
