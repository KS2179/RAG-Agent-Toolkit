"""
ingestion/parsers package init.
"""
from ingestion.parsers.base import BaseParser, ParsedDocument, UnsupportedFormatError
from ingestion.parsers.registry import get_parser_for

__all__ = ["BaseParser", "ParsedDocument", "UnsupportedFormatError", "get_parser_for"]
