"""
tests/test_phase2.py
Unit tests for Phase 2:
  - Hybrid score fusion (weighted combination)
  - BM25Index tokenisation and query
  - CrossEncoderReranker ordering
  - CitationEnforcer: valid / missing / hallucinated / retry / raise
"""
from __future__ import annotations

import math
import re
import pytest
from unittest.mock import MagicMock, patch



def _make_chunk(citation_id: str, text: str = "some text", score: float = 0.5):
    from store.vector_store import RetrievedChunk
    doc_part, idx = citation_id.split(":")
    return RetrievedChunk(
        chunk_dict={
            "text": text,
            "doc_id": doc_part * (8 // len(doc_part) + 1),
            "chunk_index": int(idx),
            "source": "test.txt",
            "citation_id": citation_id,
            "token_count": 100,
        },
        score=score,
    )


def _citation_cfg(
    enabled=True,
    min_required=1,
    on_violation="warn",
    max_retries=1,
):
    from config.loader import CitationEnforcementConfig
    return CitationEnforcementConfig(
        enabled=enabled,
        min_citations_required=min_required,
        citation_pattern=r"\[([a-f0-9]{8}:\d+)\]",
        on_violation=on_violation,
        max_retries=max_retries,
    )



class TestBM25Index:

    def _make_chunks(self):
        from ingestion.chunker import Chunk
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Retrieval augmented generation reduces hallucination",
            "Vector databases store high dimensional embeddings",
            "BM25 is a classic sparse retrieval algorithm",
        ]
        chunks = []
        for i, text in enumerate(texts):
            c = Chunk(
                text=text, doc_id=f"doc{i}" * 2, chunk_index=0,
                source=f"doc{i}.txt", token_count=len(text.split()),
                start_token=0, end_token=len(text.split()),
            )
            chunks.append(c)
        return chunks

    def test_build_and_query_returns_results(self):
        pytest.importorskip("rank_bm25")
        from retrieval.bm25_index import BM25Index
        idx = BM25Index()
        idx.build(self._make_chunks())
        results = idx.query("BM25 sparse retrieval", top_k=2)
        assert len(results) <= 2
        assert all(r.score >= 0 for r in results)

    def test_top_result_is_most_relevant(self):
        pytest.importorskip("rank_bm25")
        from retrieval.bm25_index import BM25Index
        idx = BM25Index()
        idx.build(self._make_chunks())
        results = idx.query("BM25 sparse retrieval algorithm", top_k=4)
        top_text = results[0].text.lower()
        assert "bm25" in top_text

    def test_empty_corpus_returns_empty(self):
        pytest.importorskip("rank_bm25")
        from retrieval.bm25_index import BM25Index
        idx = BM25Index()
        idx.build([])
        assert idx.query("anything", top_k=5) == []

    def test_scores_normalised_to_0_1(self):
        pytest.importorskip("rank_bm25")
        from retrieval.bm25_index import BM25Index
        idx = BM25Index()
        idx.build(self._make_chunks())
        results = idx.query("retrieval", top_k=4)
        for r in results:
            assert 0.0 <= r.score <= 1.0



class TestHybridFusion:

    def test_fused_score_is_weighted_combination(self):
        from retrieval.hybrid_retriever import HybridRetriever

        vec_chunks  = [_make_chunk("abcd1234:0", score=1.0),
                       _make_chunk("abcd1234:1", score=0.8)]
        bm25_chunks = [_make_chunk("abcd1234:0", score=0.6),
                       _make_chunk("efgh5678:0", score=0.9)]

        alpha = 0.6
        fused = HybridRetriever._fuse(vec_chunks, bm25_chunks, alpha=alpha)

        chunk_00 = next(c for c in fused if c.citation_id == "abcd1234:0")
        expected = alpha * 1.0 + (1 - alpha) * 0.6
        assert abs(chunk_00.score - expected) < 1e-6

    def test_chunk_only_in_bm25_gets_zero_vector_score(self):
        from retrieval.hybrid_retriever import HybridRetriever

        vec_chunks  = [_make_chunk("abcd1234:0", score=0.9)]
        bm25_chunks = [_make_chunk("efgh5678:0", score=0.7)]

        alpha = 0.6
        fused = HybridRetriever._fuse(vec_chunks, bm25_chunks, alpha=alpha)

        bm25_only = next(c for c in fused if c.citation_id == "efgh5678:0")
        expected = alpha * 0.0 + (1 - alpha) * 0.7
        assert abs(bm25_only.score - expected) < 1e-6

    def test_fused_result_is_sorted_descending(self):
        from retrieval.hybrid_retriever import HybridRetriever

        vec_chunks  = [_make_chunk("aaaa1111:0", score=0.3),
                       _make_chunk("bbbb2222:0", score=0.9)]
        bm25_chunks = [_make_chunk("aaaa1111:0", score=0.2),
                       _make_chunk("bbbb2222:0", score=0.8)]

        fused = HybridRetriever._fuse(vec_chunks, bm25_chunks, alpha=0.5)
        scores = [c.score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_union_of_chunks_included(self):
        from retrieval.hybrid_retriever import HybridRetriever

        vec   = [_make_chunk("aaaa1111:0")]
        bm25  = [_make_chunk("bbbb2222:0")]
        fused = HybridRetriever._fuse(vec, bm25, alpha=0.5)
        ids   = {c.citation_id for c in fused}
        assert ids == {"aaaa1111:0", "bbbb2222:0"}



class TestCrossEncoderReranker:

    def test_reranker_sorts_by_score(self):
        pytest.importorskip("sentence_transformers")
        from retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

        mock_model = MagicMock()
        import numpy as np
        mock_model.predict.return_value = np.array([0.1, 5.0, 2.3])
        reranker._model = mock_model

        candidates = [
            _make_chunk("aaaa1111:0", "low relevance"),
            _make_chunk("bbbb2222:0", "high relevance"),
            _make_chunk("cccc3333:0", "medium relevance"),
        ]
        ranked = reranker.rerank("query", candidates, top_k=3)

        assert ranked[0].citation_id == "bbbb2222:0"  
        assert len(ranked) == 3

    def test_top_k_trims_results(self):
        pytest.importorskip("sentence_transformers")
        from retrieval.reranker import CrossEncoderReranker
        import numpy as np

        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        reranker._model = mock_model

        candidates = [_make_chunk(f"{'a'*8}:{i}") for i in range(4)]
        ranked = reranker.rerank("query", candidates, top_k=2)
        assert len(ranked) == 2

    def test_scores_are_sigmoid_normalised(self):
        pytest.importorskip("sentence_transformers")
        from retrieval.reranker import CrossEncoderReranker
        import numpy as np

        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.0])
        reranker._model = mock_model

        candidates = [_make_chunk("aaaa1111:0")]
        ranked = reranker.rerank("query", candidates, top_k=1)
        assert abs(ranked[0].score - 0.5) < 1e-6   



class TestCitationEnforcer:

    def _chunks(self):
        return [
            _make_chunk("abcd1234:0"),
            _make_chunk("abcd1234:1"),
        ]

    def test_valid_answer_passes(self):
        from retrieval.citation_enforcer import CitationEnforcer
        enforcer = CitationEnforcer(_citation_cfg())
        answer   = "RAG reduces hallucination [abcd1234:0] by grounding [abcd1234:1]."
        result   = enforcer.check(answer, self._chunks())
        assert result.is_valid
        assert len(result.found_citations) == 2

    def test_missing_citation_fails(self):
        from retrieval.citation_enforcer import CitationEnforcer
        enforcer = CitationEnforcer(_citation_cfg(min_required=1))
        result   = enforcer.check("No citations here.", self._chunks())
        assert not result.is_valid
        assert result.missing_count == 1

    def test_hallucinated_citation_fails(self):
        from retrieval.citation_enforcer import CitationEnforcer
        enforcer = CitationEnforcer(_citation_cfg())
        answer   = "See [ffffffff:9] for details."   
        result   = enforcer.check(answer, self._chunks())
        assert not result.is_valid
        assert "ffffffff:9" in result.invalid_citations

    def test_on_violation_raise(self):
        from retrieval.citation_enforcer import CitationEnforcer, CitationViolationError
        enforcer = CitationEnforcer(_citation_cfg(on_violation="raise"))
        with pytest.raises(CitationViolationError):
            enforcer.enforce("No citations.", self._chunks())

    def test_on_violation_retry_succeeds(self):
        from retrieval.citation_enforcer import CitationEnforcer
        enforcer = CitationEnforcer(
            _citation_cfg(on_violation="retry", max_retries=2)
        )
        call_count = {"n": 0}

        def retry_fn(prev):
            call_count["n"] += 1
            return "Fixed [abcd1234:0]."

        result = enforcer.enforce("No citations.", self._chunks(), retry_fn=retry_fn)
        assert result.is_valid
        assert call_count["n"] == 1

    def test_enforcement_disabled_always_passes(self):
        from retrieval.citation_enforcer import CitationEnforcer
        enforcer = CitationEnforcer(_citation_cfg(enabled=False))
        result   = enforcer.enforce("No citations.", self._chunks())
        assert result.is_valid

    def test_deduplicates_same_citation(self):
        from retrieval.citation_enforcer import CitationEnforcer
        enforcer = CitationEnforcer(_citation_cfg(min_required=1))
        answer   = "[abcd1234:0] is repeated [abcd1234:0] twice."
        result   = enforcer.check(answer, self._chunks())
        assert len(result.found_citations) == 1   
        assert result.is_valid
