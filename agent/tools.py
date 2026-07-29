"""
agent/tools.py
Tool definitions and adapters for local document search.
"""
from __future__ import annotations

from typing import Any, Dict

LOCAL_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_local_documents",
        "description": (
            "Search the user's own ingested documents (PDFs, Word docs, LaTeX, notes, etc). "
            "Use this first for anything that plausibly lives in the user's private corpus. "
            "Returns ranked chunks with citation IDs and similarity scores; a low top score "
            "means the corpus likely doesn't cover this question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to match against local document chunks.",
                }
            },
            "required": ["query"],
        },
    },
}


def run_local_search(retriever: Any, query: str, score_threshold: float = 0.75) -> Dict[str, Any]:
    """
    Executes local document search via retriever, returns structured results with
    top_score and below_confidence_threshold metadata.
    """
    res = retriever.retrieve(query)
    chunks_data = [
        {
            "citation_id": chunk.citation_id,
            "source": chunk.source,
            "score": chunk.score,
            "text": chunk.text,
        }
        for chunk in res.chunks
    ]

    top_score = res.chunks[0].score if res.chunks else 0.0
    below_threshold = top_score < score_threshold

    return {
        "chunks": chunks_data,
        "top_score": top_score,
        "below_confidence_threshold": below_threshold,
    }
