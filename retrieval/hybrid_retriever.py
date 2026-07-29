"""
retrieval/hybrid_retriever.py
Phase 2 retriever.  Extends the Phase 1 Retriever with:

  1. Hybrid fusion  — combines BM25 keyword scores with dense vector scores
                      using a weighted Reciprocal Rank Fusion (RRF) variant
  2. Cross-encoder re-ranking  — rescores the fused candidate set
  3. Citation enforcement      — validates / retries answers post-generation

The public interface is identical to Phase 1 Retriever so callers need
no changes — just swap `Retriever` → `HybridRetriever`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.loader import PipelineConfig
from ingestion.embedder import BaseEmbedder
from retrieval.bm25_index import BM25Index
from retrieval.citation_enforcer import CitationEnforcer
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import RAGAnswer, RetrievalResult, Retriever
from store.vector_store import BaseVectorStore, RetrievedChunk


class HybridRetriever(Retriever):
    """
    Drop-in replacement for Phase 1 Retriever.
    Adds: hybrid fusion · re-ranking · citation enforcement.
    """

    def __init__(
        self,
        embedder:      BaseEmbedder,
        vector_store:  BaseVectorStore,
        bm25_index:    BM25Index,
        cfg:           PipelineConfig,
    ):
        super().__init__(embedder, vector_store, cfg)
        self.bm25_index = bm25_index

        rc = cfg.retrieval
        self.reranker = (
            CrossEncoderReranker(rc.reranker.model)
            if rc.reranker.enabled
            else None
        )
        self.enforcer = CitationEnforcer(cfg.citation_enforcement)

    def retrieve(self, query: str) -> RetrievalResult:
        rc = self.cfg.retrieval

        n_candidates = rc.top_k * rc.hybrid.candidate_multiplier

        query_embedding = self.embedder.embed_query(query)
        vector_chunks   = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=n_candidates,
            score_threshold=rc.score_threshold,
        )

        bm25_chunks = self.bm25_index.query(query, top_k=n_candidates)

        fused = self._fuse(
            vector_chunks,
            bm25_chunks,
            alpha=rc.hybrid.vector_weight,
        )

        if self.reranker is not None:
            final_chunks = self.reranker.rerank(
                query=query,
                candidates=fused,
                top_k=rc.reranker.top_k_after_rerank,
            )
        else:
            final_chunks = fused[: rc.top_k]

        context_block, citations = self._format_context(final_chunks)

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            context_block=context_block,
            citations=citations,
        )

    def build_retry_prompt(
        self,
        result: RetrievalResult,
        previous_answer: str,
    ) -> Tuple[str, str]:
        prompts = self.cfg.prompts
        retry_template = prompts.rag_retry or prompts.rag_user

        user_prompt = retry_template.format(
            context=result.context_block,
            question=result.query,
            previous_answer=previous_answer,
        )
        return prompts.rag_system, user_prompt

    @staticmethod
    def _fuse(
        vector_chunks: List[RetrievedChunk],
        bm25_chunks:   List[RetrievedChunk],
        alpha: float,
    ) -> List[RetrievedChunk]:
        """
        Weighted score fusion.

        For each chunk that appears in either result list:
            fused_score = alpha * vector_score + (1 - alpha) * bm25_score

        Chunks that appear in only one list receive a score of 0 for the
        missing modality (conservative — they were not retrieved by that method).
        """
        vec_map:  Dict[str, Tuple[RetrievedChunk, float]] = {
            c.citation_id: (c, c.score) for c in vector_chunks
        }
        bm25_map: Dict[str, float] = {
            c.citation_id: c.score for c in bm25_chunks
        }

        all_ids = set(vec_map.keys()) | set(bm25_map.keys())

        fused: List[Tuple[float, RetrievedChunk]] = []
        for cid in all_ids:
            if cid in vec_map:
                chunk, vec_score = vec_map[cid]
            else:
                bm25_chunk = next(c for c in bm25_chunks if c.citation_id == cid)
                chunk, vec_score = bm25_chunk, 0.0

            bm25_score = bm25_map.get(cid, 0.0)
            score = alpha * vec_score + (1.0 - alpha) * bm25_score
            chunk.score = score
            fused.append((score, chunk))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in fused]


class HybridRAGPipeline:
    """
    Phase 2 full pipeline:
        HybridRetriever → LLM generation → CitationEnforcer (with retry)
    """

    def __init__(self, retriever: HybridRetriever, cfg: PipelineConfig):
        self.retriever = retriever
        self.cfg       = cfg
        self._llm      = self._init_llm()

    def _init_llm(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            return None

    def answer(self, question: str, model: str = "gpt-4o-mini") -> RAGAnswer:
        if self._llm is None:
            raise RuntimeError("LLM client not initialised. Check OPENAI_API_KEY.")

        result = self.retriever.retrieve(question)
        system, user = self.retriever.build_prompt(result)

        def _call_llm(system_p: str, user_p: str) -> str:
            resp = self._llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user",   "content": user_p},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content

        answer_text = _call_llm(system, user)

        def _retry_fn(prev_answer: str) -> str:
            retry_system, retry_user = self.retriever.build_retry_prompt(
                result, prev_answer
            )
            return _call_llm(retry_system, retry_user)

        check = self.retriever.enforcer.enforce(
            answer=answer_text,
            retrieved_chunks=result.chunks,
            retry_fn=_retry_fn,
        )

        return RAGAnswer(
            question=question,
            answer=check.answer,
            retrieved_chunks=result.chunks,
            citations=result.citations,
            context_block=result.context_block,
        )
