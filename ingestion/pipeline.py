"""
ingestion/pipeline.py
Orchestrates the full ingestion flow:
    raw documents → chunks → embeddings → vector store → BM25 index

Re-ingestion is idempotent — chunk IDs are content-hashed.
Calling build_ingestion_pipeline("config/phase2.yaml") wires up the
HybridRetriever automatically; "config/phase1.yaml" still returns the
plain Phase 1 Retriever — no caller changes needed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from config.loader import PipelineConfig, load_config
from ingestion.chunker import Chunk, TokenChunker
from ingestion.embedder import BaseEmbedder, build_embedder
from store.vector_store import BaseVectorStore, build_vector_store


class IngestionStats:
    def __init__(self, num_documents: int, num_chunks: int, elapsed_seconds: float, skipped: Optional[List[dict]] = None):
        self.num_documents   = num_documents
        self.num_chunks      = num_chunks
        self.elapsed_seconds = elapsed_seconds
        self.skipped         = skipped or []

    def __str__(self) -> str:
        msg = (
            f"Ingested {self.num_documents} document(s) → "
            f"{self.num_chunks} chunks in {self.elapsed_seconds:.1f}s"
        )
        if self.skipped:
            msg += f" ({len(self.skipped)} skipped)"
        return msg


class IngestionPipeline:
    """
    Wires together:
      TokenChunker → BaseEmbedder → BaseVectorStore (→ BM25Index if Phase 2)

    All components are injected so they can be mocked in tests.
    """

    def __init__(
        self,
        chunker:      TokenChunker,
        embedder:     BaseEmbedder,
        vector_store: BaseVectorStore,
        cfg:          PipelineConfig,
        bm25_index=None,    
    ):
        self.chunker      = chunker
        self.embedder     = embedder
        self.vector_store = vector_store
        self.cfg          = cfg
        self.bm25_index   = bm25_index

    def ingest_documents(self, documents: List[dict]) -> IngestionStats:
        """
        Ingest a list of document dicts.

        Each dict must contain:
          - "text"     (str)  – full document content
          - "source"   (str)  – filename, URL, or human label
          - "metadata" (dict, optional)
        """
        t0 = time.time()

        chunks: List[Chunk] = self.chunker.chunk_documents(documents)
        if not chunks:
            return IngestionStats(len(documents), 0, time.time() - t0)

        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.upsert(chunks, embeddings)

        if self.bm25_index is not None:
            self.bm25_index.build(self._all_chunks())

        return IngestionStats(
            num_documents=len(documents),
            num_chunks=len(chunks),
            elapsed_seconds=time.time() - t0,
        )

    def _all_chunks(self) -> List[Chunk]:
        """Pull every stored chunk from Chroma for BM25 index rebuilding."""
        from retrieval.bm25_index import _dict_to_chunk
        raw = self.vector_store._collection.get(include=["documents", "metadatas"])
        return [
            _dict_to_chunk({"text": doc, **meta})
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]

    def ingest_file(self, path: str, metadata: Optional[dict] = None) -> IngestionStats:
        from ingestion.parsers.registry import get_parser_for
        p = Path(path)
        keep_math = getattr(getattr(getattr(self.cfg, "ingestion", None), "parsers", None), "latex", None)
        keep_math_val = getattr(keep_math, "keep_math", False) if keep_math else False
        parser = get_parser_for(p, keep_math=keep_math_val)
        parsed = parser.parse(p)
        merged_meta = {**parsed.metadata, **(metadata or {})}
        return self.ingest_documents([{
            "text": parsed.text, "source": parsed.source, "metadata": merged_meta,
        }])

    def ingest_directory(self, directory: str, glob: str = "**/*") -> IngestionStats:
        from ingestion.parsers.base import UnsupportedFormatError
        from ingestion.parsers.registry import get_parser_for
        d = Path(directory)
        docs = []
        skipped = []
        keep_math = getattr(getattr(getattr(self.cfg, "ingestion", None), "parsers", None), "latex", None)
        keep_math_val = getattr(keep_math, "keep_math", False) if keep_math else False

        for p in d.glob(glob):
            if not p.is_file():
                continue
            try:
                parser = get_parser_for(p, keep_math=keep_math_val)
                parsed = parser.parse(p)
                docs.append({
                    "text": parsed.text,
                    "source": str(p.relative_to(d)),
                    "metadata": {**parsed.metadata, "filepath": str(p)},
                })
            except UnsupportedFormatError as e:
                skipped.append({"file": str(p), "reason": str(e)})
            except Exception as e:
                skipped.append({"file": str(p), "reason": str(e)})

        stats = self.ingest_documents(docs)
        stats.skipped = skipped
        return stats


class RetrieverBundle:
    """Convenience holder returned by build_ingestion_pipeline."""
    def __init__(self, retriever, rag):
        self.retriever = retriever
        self.rag       = rag


def build_ingestion_pipeline(
    config_path: str = "config/phase1.yaml",
) -> tuple[IngestionPipeline, RetrieverBundle]:
    """
    Build and return both ingestion and retrieval components from one config.

    Phase 1 YAML  →  plain vector Retriever + RAGPipeline
    Phase 2 YAML  →  HybridRetriever + HybridRAGPipeline  (auto-detected)
    """
    cfg          = load_config(config_path)
    chunker      = TokenChunker(cfg.chunking)
    embedder     = build_embedder(cfg.embedding)
    vector_store = build_vector_store(cfg.vector_store)

    if cfg.retrieval.hybrid.enabled:
        from retrieval.bm25_index import BM25Index
        from retrieval.hybrid_retriever import HybridRAGPipeline, HybridRetriever

        bm25_index = BM25Index()
        pipeline   = IngestionPipeline(
            chunker, embedder, vector_store, cfg, bm25_index=bm25_index
        )
        retriever = HybridRetriever(embedder, vector_store, bm25_index, cfg)
        rag       = HybridRAGPipeline(retriever, cfg)
    else:
        from retrieval.retriever import RAGPipeline, Retriever

        pipeline  = IngestionPipeline(chunker, embedder, vector_store, cfg)
        retriever = Retriever(embedder, vector_store, cfg)
        rag       = RAGPipeline(retriever, cfg)

    return pipeline, RetrieverBundle(retriever=retriever, rag=rag)
