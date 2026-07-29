"""
ingestion/parsers/docx_parser.py
Parser for Microsoft Word (.docx) documents using python-docx.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ingestion.parsers.base import BaseParser, ParsedDocument


class DocxParser(BaseParser):
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("docx",)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx package is required for DOCX parsing. Install it via 'pip install python-docx'."
            )

        doc = docx.Document(str(path))
        text_blocks = []

        # Extract text from paragraphs and table cells in document order
        for element in doc.element.body:
            if element.tag.endswith("p"):
                para = docx.text.paragraph.Paragraph(element, doc)
                if para.text.strip():
                    text_blocks.append(para.text.strip())
            elif element.tag.endswith("tbl"):
                table = docx.table.Table(element, doc)
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        text_blocks.append(" | ".join(row_cells))

        full_text = "\n\n".join(text_blocks)

        return ParsedDocument(
            text=full_text,
            source=path.name,
            metadata={"format": "docx"},
        )
