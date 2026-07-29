"""
mcp_server/server.py
Custom FastMCP server exposing the project's RAG pipeline as reusable MCP tools.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.loader import load_config
from ingestion.pipeline import build_ingestion_pipeline


def _resolve_within_allowlist(given_path: str, allowlist_root: str) -> Path:
    """
    Resolves given_path and verifies it resides inside allowlist_root (after symlink resolution).
    Raises ValueError if path attempts traversal outside allowlist_root.
    """
    root = Path(allowlist_root).resolve()
    target = Path(given_path).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path traversal rejected: '{given_path}' resolves outside the allowed root directory '{root}'."
        )

    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")

    return target


def create_server(config_path: str = "config/phase4.yaml") -> Any:
    """
    Factory creating a FastMCP server instance wired to the RAG pipeline.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("mcp package is required. Install via 'pip install mcp'.")

    cfg = load_config(config_path)
    pipeline, bundle = build_ingestion_pipeline(config_path)
    allowlist_root = cfg.mcp.server.ingest_allowlist_root

    mcp = FastMCP("rag-pipeline")

    @mcp.tool()
    def rag_query(question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using the ingested document corpus.
        Returns answer text, inline citations, and retrieved chunks.
        """
        if top_k is not None:
            bundle.retriever.cfg.retrieval.top_k = top_k
        answer = bundle.rag.answer(question)
        return {
            "answer": answer.answer,
            "citations": answer.citations,
            "chunks": [
                {
                    "citation_id": c.citation_id,
                    "source": c.source,
                    "score": c.score,
                    "text": c.text,
                }
                for c in answer.retrieved_chunks
            ],
        }

    @mcp.tool()
    def ingest_document(path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest a document into the corpus.
        'path' must resolve inside the configured allow-listed root.
        """
        safe_path = _resolve_within_allowlist(path, allowlist_root)
        stats = pipeline.ingest_file(str(safe_path), metadata=metadata)
        return {
            "num_documents": stats.num_documents,
            "num_chunks": stats.num_chunks,
            "elapsed_seconds": stats.elapsed_seconds,
            "skipped": stats.skipped,
        }

    @mcp.tool()
    def list_ingested_sources() -> List[Dict[str, Any]]:
        """
        List all sources currently ingested in the vector store collection.
        """
        raw = pipeline.vector_store._collection.get(include=["metadatas"])
        sources_seen = set()
        sources_list = []
        for meta in raw.get("metadatas", []):
            src = meta.get("source", "unknown")
            if src not in sources_seen:
                sources_seen.add(src)
                sources_list.append({
                    "source": src,
                    "doc_id": meta.get("doc_id", ""),
                    "format": meta.get("format", ""),
                })
        return sources_list

    return mcp
