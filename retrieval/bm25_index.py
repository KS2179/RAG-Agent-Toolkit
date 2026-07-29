"""
retrieval/bm25_index.py
Sparse keyword search using BM25 (Okapi BM25 via rank-bm25).

The index is built in-memory from the chunks already stored in the vector
store.  It is rebuilt on each process start (cheap — milliseconds for
thousands of chunks) so there is no separate persistence concern.

Interface mirrors BaseVectorStore.query() so the hybrid layer can treat
both retrievers identically.
"""
from __future__ import annotations

import re
import string
from typing import List, Optional

from ingestion.chunker import Chunk
from store.vector_store import RetrievedChunk


class BM25Index:
    """
    In-memory BM25 index over a fixed corpus of chunks.

    Build once after ingestion; rebuild whenever new docs are added.
    """

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._bm25 = None           

    def build(self, chunks: List[Chunk]) -> None:
        """
        (Re)build the BM25 index over the given chunks.
        Call this after every ingestion run.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: pip install rank-bm25")

        self._chunks = chunks
        tokenised_corpus = [self._tokenise(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenised_corpus)

    def build_from_vector_store(self, vector_store) -> None:
        """
        Pull all chunks out of a Chroma vector store and build the index.
        This is the convenience path — no need to pass chunks separately.
        """
        raw = vector_store._collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(raw["documents"], raw["metadatas"]):
            chunk_dict = {"text": doc, **meta}
            chunks.append(_dict_to_chunk(chunk_dict))
        self.build(chunks)

    def query(
        self,
        query_text: str,
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Return the top_k chunks ranked by BM25 score.
        Scores are normalised to [0, 1] relative to the highest score in
        this query so they can be combined with cosine similarity scores.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokenised_query = self._tokenise(query_text)
        raw_scores = self._bm25.get_scores(tokenised_query)

        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        norm_scores = [s / max_score for s in raw_scores]

        ranked = sorted(
            enumerate(norm_scores), key=lambda x: x[1], reverse=True
        )

        results: List[RetrievedChunk] = []
        for idx, score in ranked[:top_k]:
            if score < score_threshold:
                break
            chunk = self._chunks[idx]
            results.append(RetrievedChunk(chunk.to_dict(), score=score))

        return results

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Lowercase, strip punctuation, split on whitespace.
        Deliberately simple — fast and sufficient for BM25 token matching.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()


def _dict_to_chunk(d: dict) -> Chunk:
    from ingestion.chunker import Chunk
    return Chunk(
        text        = d.get("text", ""),
        doc_id      = d.get("doc_id", ""),
        chunk_index = int(d.get("chunk_index", 0)),
        source      = d.get("source", ""),
        token_count = int(d.get("token_count", 0)),
        start_token = int(d.get("start_token", 0)),
        end_token   = int(d.get("end_token", 0)),
        metadata    = {
            k: v for k, v in d.items()
            if k not in {"text", "doc_id", "chunk_index", "source",
                         "token_count", "start_token", "end_token", "citation_id"}
        },
    )
