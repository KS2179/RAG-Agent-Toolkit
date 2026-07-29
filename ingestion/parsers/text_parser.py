"""
ingestion/parsers/text_parser.py
Parser for plain text (.txt) and Markdown (.md) files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ingestion.parsers.base import BaseParser, ParsedDocument


class TextParser(BaseParser):
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("txt", "md")

    def parse(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        ext = path.suffix.lower().lstrip(".")
        return ParsedDocument(
            text=text,
            source=path.name,
            metadata={"format": ext},
        )
