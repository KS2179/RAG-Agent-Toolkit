"""
ingestion/parsers/latex_parser.py
Parser for LaTeX (.tex) files using pylatexenc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ingestion.parsers.base import BaseParser, ParsedDocument


class LatexParser(BaseParser):
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("tex",)

    def __init__(self, keep_math: bool = False):
        self.keep_math = keep_math

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from pylatexenc.latex2text import LatexNodes2Text
        except ImportError:
            raise ImportError(
                "pylatexenc package is required for LaTeX parsing. Install it via 'pip install pylatexenc'."
            )

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        converter = LatexNodes2Text(keep_math=self.keep_math)
        converted_text = converter.latex_to_text(raw_text).strip()

        return ParsedDocument(
            text=converted_text,
            source=path.name,
            metadata={"format": "tex"},
        )
