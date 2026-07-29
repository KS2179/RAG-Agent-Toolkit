"""
retrieval/retriever.py
Pulls the top-k most relevant chunks for a query and packages them with
inline citations ready to inject into the LLM prompt.

Phase 2 will extend this class with hybrid BM25 + vector search and
a cross-encoder re-ranker — the interface stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from config.loader import PipelineConfig
from ingestion.embedder import BaseEmbedder
from store.vector_store import BaseVectorStore, RetrievedChunk



@dataclass
class RetrievalResult:
    """Everything the generation step needs."""
    query: str
    chunks: List[RetrievedChunk]        
    context_block: str                  
    citations: List[str]                

class Retriever:
    """
    Phase 1: pure vector (semantic) retrieval.
    Phase 2: will be subclassed / extended with BM25 + re-ranking.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        cfg: PipelineConfig,
    ):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.cfg          = cfg

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Embed the query → search the vector store → return ranked chunks
        wrapped with a formatted context block and citation list.
        """
        retrieval_cfg = self.cfg.retrieval

        query_embedding = self.embedder.embed_query(query)

        chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=retrieval_cfg.top_k,
            score_threshold=retrieval_cfg.score_threshold,
        )

        context_block, citations = self._format_context(chunks)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            context_block=context_block,
            citations=citations,
        )

    def build_prompt(self, result: RetrievalResult) -> tuple[str, str]:
        """
        Fill the versioned prompt templates from config/phase1.yaml.
        Returns (system_prompt, user_prompt).
        """
        prompts = self.cfg.prompts
        user_prompt = prompts.rag_user.format(
            context=result.context_block,
            question=result.query,
        )
        return prompts.rag_system, user_prompt

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> tuple[str, List[str]]:
        """
        Build the context block injected into the prompt.

        Format:
        ──────────────────────────────────────
        [a1b2c3d4:0]  (source.txt, score=0.91)
        <chunk text>

        [a1b2c3d4:1]  (source.txt, score=0.87)
        <chunk text>
        ──────────────────────────────────────
        """
        parts: List[str] = []
        citations: List[str] = []

        for chunk in chunks:
            citation = f"[{chunk.citation_id}]"
            citations.append(citation)
            header = f"{citation}  (source: {chunk.source}, score: {chunk.score:.4f})"
            parts.append(f"{header}\n{chunk.text}")

        context_block = "\n\n".join(parts)
        return context_block, citations


class RAGPipeline:
    """
    Thin wrapper that chains Retriever → OpenAI completion.
    Keeps generation logic separate from retrieval so each can be
    tested and swapped independently.
    """

    def __init__(
        self,
        retriever: Retriever,
        cfg: PipelineConfig,
    ):
        self.retriever = retriever
        self.cfg       = cfg
        self._llm      = self._init_llm()

    def _init_llm(self):
        try:
            from openai import OpenAI
            import os
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            return None  

    def answer(self, question: str, model: str = "gpt-4o-mini") -> "RAGAnswer":
        """Full RAG cycle: retrieve → build prompt → generate → return answer."""
        result          = self.retriever.retrieve(question)
        system_prompt, user_prompt = self.retriever.build_prompt(result)

        if self._llm is None:
            raise RuntimeError(
                "LLM client not initialised. Check OPENAI_API_KEY."
            )

        response = self._llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,    
        )

        answer_text = response.choices[0].message.content

        return RAGAnswer(
            question=question,
            answer=answer_text,
            retrieved_chunks=result.chunks,
            citations=result.citations,
            context_block=result.context_block,
        )


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[str]
    context_block: str

    def pretty_print(self) -> None:
        """Print a formatted answer to stdout."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            console = Console()
            console.print(Panel(self.answer, title="Answer", border_style="green"))
            console.print("\n[bold]Sources used:[/bold]")
            for chunk in self.retrieved_chunks:
                console.print(
                    f"  • [{chunk.citation_id}]  {chunk.source}  "
                    f"(score={chunk.score:.4f})"
                )
        except ImportError:
            print(f"Answer:\n{self.answer}")
            print("\nSources:")
            for chunk in self.retrieved_chunks:
                print(f"  [{chunk.citation_id}] {chunk.source}")
