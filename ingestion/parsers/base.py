"""
ingestion/parsers/base.py
Base abstractions for multi-format document parsers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ParsedDocument:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


class UnsupportedFormatError(Exception):
    """Raised when no parser can handle a file format."""
    pass


class BaseParser:
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ()

    def parse(self, path: Path) -> ParsedDocument:
        raise NotImplementedError

    @classmethod
    def can_parse(cls, path: Path, sniffed_type: Optional[str] = None) -> bool:
        ext = path.suffix.lower().lstrip(".")
        return ext in cls.SUPPORTED_EXTENSIONS
