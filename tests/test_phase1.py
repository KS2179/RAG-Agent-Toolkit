"""
tests/test_phase1.py
Unit tests for Phase 1:
  - TokenChunker produces chunks within token bounds
  - Chunk metadata is correct (doc_id, citation_id, overlap)
  - IngestionPipeline wires together correctly (mocked embedder + store)
  - Retriever returns ranked results with citation strings
"""
from __future__ import annotations

import pytest
from typing import List
from unittest.mock import MagicMock, patch


@pytest.fixture
def chunking_cfg():
    from config.loader import ChunkingConfig
    return ChunkingConfig(min_tokens=100, max_tokens=150, overlap_tokens=20)


@pytest.fixture
def sample_text():
    return " ".join(["word"] * 600)


@pytest.fixture
def sample_documents():
    return [
        {"text": " ".join(["alpha"] * 600), "source": "doc_a.txt"},
        {"text": " ".join(["beta"] * 400),  "source": "doc_b.txt"},
    ]



class TestTokenChunker:

    def test_chunks_within_token_bounds(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        chunks = chunker.chunk_document(sample_text, source="test.txt")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count <= chunking_cfg.max_tokens, (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens "
                f"(max={chunking_cfg.max_tokens})"
            )

    def test_all_chunks_except_last_above_min(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        chunks = chunker.chunk_document(sample_text, source="test.txt")

        for chunk in chunks[:-1]:
            assert chunk.token_count >= chunking_cfg.min_tokens

    def test_chunk_indices_are_sequential(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document(sample_text, "t.txt")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_citation_id_format(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document(sample_text, "t.txt")
        for chunk in chunks:
            parts = chunk.citation_id.split(":")
            assert len(parts) == 2
            assert len(parts[0]) == 8
            assert parts[1].isdigit()

    def test_same_text_same_doc_id(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        c1 = chunker.chunk_document(sample_text, "a.txt")
        c2 = chunker.chunk_document(sample_text, "b.txt")
        assert c1[0].doc_id == c2[0].doc_id  # content hash, not source

    def test_empty_text_returns_no_chunks(self, chunking_cfg):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document("", "empty.txt")
        assert chunks == []

    def test_extra_metadata_propagated(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document(
            sample_text, "t.txt", extra_metadata={"author": "alice"}
        )
        for chunk in chunks:
            assert chunk.metadata.get("author") == "alice"

    def test_overlap_means_tokens_shared(self, chunking_cfg, sample_text):
        """Consecutive chunks should share exactly overlap_tokens tokens."""
        from ingestion.chunker import TokenChunker
        import tiktoken

        chunker = TokenChunker(chunking_cfg)
        enc = tiktoken.get_encoding(chunking_cfg.tokenizer)
        full_tokens = enc.encode(sample_text)

        chunks = chunker.chunk_document(sample_text, "t.txt")
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        c0_end = chunks[0].end_token
        c1_start = chunks[1].start_token
        overlap = c0_end - c1_start
        assert overlap == chunking_cfg.overlap_tokens



class TestIngestionPipeline:

    def _build_pipeline(self, chunking_cfg):
        from ingestion.chunker import TokenChunker
        from ingestion.pipeline import IngestionPipeline
        from config.loader import (
            EmbeddingConfig, VectorStoreConfig, RetrievalConfig,
            PromptsConfig, PipelineConfig,
        )

        cfg = PipelineConfig(
            version="test",
            chunking=chunking_cfg,
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(),
            prompts=PromptsConfig(
                rag_system="You are a helpful assistant.",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )
        mock_embedder     = MagicMock()
        mock_vector_store = MagicMock()

        mock_embedder.embed_chunks.side_effect = (
            lambda chunks: [[0.1] * 1536] * len(chunks)
        )

        pipeline = IngestionPipeline(
            chunker=TokenChunker(chunking_cfg),
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            cfg=cfg,
        )
        return pipeline, mock_embedder, mock_vector_store

    def test_ingest_calls_upsert(self, chunking_cfg, sample_documents):
        pipeline, _, mock_store = self._build_pipeline(chunking_cfg)
        stats = pipeline.ingest_documents(sample_documents)
        mock_store.upsert.assert_called_once()
        assert stats.num_documents == 2
        assert stats.num_chunks > 0

    def test_ingest_empty_document_list(self, chunking_cfg):
        pipeline, _, mock_store = self._build_pipeline(chunking_cfg)
        stats = pipeline.ingest_documents([])
        mock_store.upsert.assert_not_called()
        assert stats.num_chunks == 0

    def test_embedder_called_with_chunks(self, chunking_cfg, sample_documents):
        pipeline, mock_embedder, _ = self._build_pipeline(chunking_cfg)
        pipeline.ingest_documents(sample_documents)
        mock_embedder.embed_chunks.assert_called_once()



class TestRetriever:

    def _make_retrieved_chunk(self, citation_id: str, text: str, score: float):
        from store.vector_store import RetrievedChunk
        return RetrievedChunk(
            chunk_dict={
                "text": text,
                "doc_id": citation_id.split(":")[0] * 8,
                "chunk_index": int(citation_id.split(":")[1]),
                "source": "test.txt",
                "citation_id": citation_id,
                "token_count": 120,
            },
            score=score,
        )

    def test_retrieve_returns_result_with_citations(self):
        from retrieval.retriever import Retriever
        from config.loader import (
            ChunkingConfig, EmbeddingConfig, VectorStoreConfig,
            RetrievalConfig, PromptsConfig, PipelineConfig,
        )

        cfg = PipelineConfig(
            version="test",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(top_k=3),
            prompts=PromptsConfig(
                rag_system="system",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )

        mock_embedder     = MagicMock()
        mock_vector_store = MagicMock()

        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_vector_store.query.return_value = [
            self._make_retrieved_chunk("abcd1234:0", "Relevant text.", 0.92),
            self._make_retrieved_chunk("abcd1234:1", "More context.",  0.87),
        ]

        retriever = Retriever(mock_embedder, mock_vector_store, cfg)
        result = retriever.retrieve("What is relevant?")

        assert len(result.chunks) == 2
        assert len(result.citations) == 2
        assert "[abcd1234:0]" in result.context_block
        assert result.query == "What is relevant?"

    def test_build_prompt_uses_config_templates(self):
        from retrieval.retriever import Retriever
        from config.loader import (
            ChunkingConfig, EmbeddingConfig, VectorStoreConfig,
            RetrievalConfig, PromptsConfig, PipelineConfig,
        )

        cfg = PipelineConfig(
            version="test",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(),
            prompts=PromptsConfig(
                rag_system="SYSTEM_PROMPT",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )

        mock_embedder     = MagicMock()
        mock_vector_store = MagicMock()
        mock_embedder.embed_query.return_value = [0.0] * 1536
        mock_vector_store.query.return_value   = []

        retriever = Retriever(mock_embedder, mock_vector_store, cfg)
        result    = retriever.retrieve("test question")
        system, user = retriever.build_prompt(result)

        assert system == "SYSTEM_PROMPT"
        assert "test question" in user
