"""
ingestion/parsers/registry.py
Pluggable parser registry with extension & magic-byte fallback detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

from ingestion.parsers.base import BaseParser, UnsupportedFormatError
from ingestion.parsers.text_parser import TextParser


def _get_parser_class_map() -> Dict[str, str]:
    """
    Returns mapping from file extension (lowercase) to module and class name
    for lazy loading to prevent hard import failures if optional deps are missing.
    """
    return {
        "txt": "ingestion.parsers.text_parser.TextParser",
        "md": "ingestion.parsers.text_parser.TextParser",
        "pdf": "ingestion.parsers.pdf_parser.PDFParser",
        "docx": "ingestion.parsers.docx_parser.DocxParser",
        "tex": "ingestion.parsers.latex_parser.LatexParser",
    }


def get_parser_for(path: Path, keep_math: bool = False) -> BaseParser:
    """
    Returns an instantiated parser suitable for the given path.
    Checks file extension first, then falls back to magic-byte sniffing via `filetype`.

    Raises UnsupportedFormatError if no parser supports the file format.
    """
    ext = path.suffix.lower().lstrip(".")
    class_map = _get_parser_class_map()

    target_class_path = class_map.get(ext)

    # Magic-byte fallback if extension missing or unknown
    if not target_class_path:
        try:
            import filetype
            kind = filetype.guess(str(path))
            if kind is not None:
                sniffed_ext = kind.extension.lower()
                target_class_path = class_map.get(sniffed_ext)
        except Exception:
            pass

    if not target_class_path:
        raise UnsupportedFormatError(
            f"Unsupported file format for '{path.name}'. Extension '{ext}' is not supported."
        )

    # Lazy import parser class
    module_name, class_name = target_class_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_name)
    cls: Type[BaseParser] = getattr(mod, class_name)

    if cls.__name__ == "LatexParser":
        return cls(keep_math=keep_math)
    return cls()
