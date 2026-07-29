"""
tests/test_phase4.py
Unit and integration tests for Phase 4:
  - Multi-format ingestion parsers (txt, md, pdf, docx, tex)
  - Parser registry & magic byte fallback
  - Ingestion directory wide glob & skipped files tracking
  - Agentic orchestrator & routing strategies
  - MCP Client Manager & namespacing
  - Custom FastMCP server & path-traversal security check
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.loader import (
    AgentConfig,
    ChunkingConfig,
    CitationEnforcementConfig,
    EmbeddingConfig,
    IngestionConfig,
    MCPConfig,
    MCPOwnServerConfig,
    MCPServerConnectionConfig,
    PipelineConfig,
    PromptsConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from ingestion.parsers.base import BaseParser, ParsedDocument, UnsupportedFormatError
from ingestion.parsers.docx_parser import DocxParser
from ingestion.parsers.latex_parser import LatexParser
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.registry import get_parser_for
from ingestion.parsers.text_parser import TextParser
from store.vector_store import RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "sample.pdf"
    if not p.exists():
        # Create minimal PDF file if pypdf or PyPDF2 is available, otherwise mock
        try:
            import pypdf
            writer = pypdf.PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with open(p, "wb") as f:
                writer.write(f)
        except Exception:
            p.write_bytes(b"%PDF-1.4 sample pdf content")
    return p


@pytest.fixture
def sample_docx_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "sample.docx"
    if not p.exists():
        try:
            import docx
            doc = docx.Document()
            doc.add_paragraph("Sample Word paragraph text.")
            doc.save(str(p))
        except Exception:
            p.write_bytes(b"PK\x03\x04 sample docx content")
    return p


@pytest.fixture
def sample_tex_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.tex"


@pytest.fixture
def phase4_cfg() -> PipelineConfig:
    return PipelineConfig(
        version="4.0.0",
        chunking=ChunkingConfig(min_tokens=50, max_tokens=100, overlap_tokens=10),
        embedding=EmbeddingConfig(),
        vector_store=VectorStoreConfig(),
        retrieval=RetrievalConfig(top_k=3, score_threshold=0.75),
        citation_enforcement=CitationEnforcementConfig(enabled=True, min_citations_required=1),
        prompts=PromptsConfig(
            rag_system="rag system",
            rag_user="Context:\n{context}\n\nQuestion: {question}",
            agent_system="agent system prompt",
        ),
        ingestion=IngestionConfig(),
        agent=AgentConfig(enabled=True, strategy="auto", max_tool_iterations=4, confidence_threshold=0.75),
        mcp=MCPConfig(
            servers=[
                MCPServerConnectionConfig(name="web_search", enabled=True, command="npx", args=["-y", "tavily-mcp"])
            ],
            server=MCPOwnServerConfig(ingest_allowlist_root="./ingest_data"),
        ),
    )


# ── Test Parsers ──────────────────────────────────────────────────────────────

class TestParsers:
    def test_text_parser(self, fixtures_dir: Path):
        parser = TextParser()
        txt_doc = parser.parse(fixtures_dir / "sample.txt")
        assert "Plain text file" in txt_doc.text
        assert txt_doc.metadata["format"] == "txt"

        md_doc = parser.parse(fixtures_dir / "sample.md")
        assert "Markdown Sample Document" in md_doc.text
        assert md_doc.metadata["format"] == "md"

    def test_latex_parser(self, sample_tex_path: Path):
        pytest.importorskip("pylatexenc")
        parser = LatexParser()
        parsed = parser.parse(sample_tex_path)
        assert "Sample LaTeX Document" in parsed.text
        assert r"\documentclass" not in parsed.text
        assert parsed.metadata["format"] == "tex"

    def test_pdf_parser_missing_dep(self, monkeypatch, sample_pdf_path: Path):
        parser = PDFParser()
        monkeypatch.setattr("builtins.__import__", MagicMock(side_effect=ImportError("No module pypdf")))
        with pytest.raises(ImportError) as exc_info:
            parser.parse(sample_pdf_path)
        assert "pypdf" in str(exc_info.value)

    def test_docx_parser_missing_dep(self, monkeypatch, sample_docx_path: Path):
        parser = DocxParser()
        monkeypatch.setattr("builtins.__import__", MagicMock(side_effect=ImportError("No module docx")))
        with pytest.raises(ImportError) as exc_info:
            parser.parse(sample_docx_path)
        assert "python-docx" in str(exc_info.value)

    def test_parser_registry(self, fixtures_dir: Path):
        parser_txt = get_parser_for(fixtures_dir / "sample.txt")
        assert isinstance(parser_txt, TextParser)

        parser_tex = get_parser_for(fixtures_dir / "sample.tex")
        assert isinstance(parser_tex, LatexParser)

        with pytest.raises(UnsupportedFormatError):
            get_parser_for(Path("unknown_file.xyz_unsupported"))

    def test_scanned_pdf_returns_empty_text(self, tmp_path: Path):
        """Scanned PDF with 0 extractable text should return ParsedDocument with empty text."""
        pytest.importorskip("pypdf")
        import pypdf
        pdf_file = tmp_path / "scanned.pdf"
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        parser = PDFParser()
        doc = parser.parse(pdf_file)
        assert doc.text == ""
        assert doc.metadata["page_count"] == 1


# ── Test Ingestion Multi-Format ───────────────────────────────────────────────

class TestIngestionMultiFormat:
    def test_ingest_directory_skips_unsupported(self, tmp_path: Path, phase4_cfg: PipelineConfig):
        from ingestion.chunker import TokenChunker
        from ingestion.pipeline import IngestionPipeline

        (tmp_path / "doc1.txt").write_text("Hello text file context.")
        (tmp_path / "doc2.unknown_ext").write_text("Binary or unknown file.")

        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_embedder.embed_chunks.side_effect = lambda chunks: [[0.1] * 1536] * len(chunks)

        pipeline = IngestionPipeline(
            chunker=TokenChunker(phase4_cfg.chunking),
            embedder=mock_embedder,
            vector_store=mock_store,
            cfg=phase4_cfg,
        )

        stats = pipeline.ingest_directory(str(tmp_path))
        assert stats.num_documents == 1
        assert len(stats.skipped) == 1
        assert "doc2.unknown_ext" in stats.skipped[0]["file"]


# ── Test Citation Enforcer Agentic ───────────────────────────────────────────

class TestAgenticCitationEnforcement:
    def _make_retrieved_chunk(self, cid: str) -> RetrievedChunk:
        return RetrievedChunk(
            {
                "text": "sample text",
                "doc_id": cid.split(":")[0],
                "chunk_index": int(cid.split(":")[1]),
                "source": "source.txt",
                "citation_id": cid,
            },
            score=0.9,
        )

    def test_mixed_citations_valid(self, phase4_cfg: PipelineConfig):
        from retrieval.citation_enforcer import CitationEnforcer

        enforcer = CitationEnforcer(phase4_cfg.citation_enforcement)
        chunks = [self._make_retrieved_chunk("abcd1234:0")]
        web_sources = [{"url": "https://example.com"}]

        answer = "Local claim [abcd1234:0] and web claim [web:1]."
        res = enforcer.check_agentic(answer, chunks, web_sources)
        assert res.is_valid
        assert len(res.found_citations) == 2

    def test_hallucinated_web_citation_fails(self, phase4_cfg: PipelineConfig):
        from retrieval.citation_enforcer import CitationEnforcer

        enforcer = CitationEnforcer(phase4_cfg.citation_enforcement)
        chunks = [self._make_retrieved_chunk("abcd1234:0")]
        web_sources = [{"url": "https://example.com"}]

        answer = "Local claim [abcd1234:0] and bad web claim [web:5]."
        res = enforcer.check_agentic(answer, chunks, web_sources)
        assert not res.is_valid
        assert "web:5" in res.invalid_citations


# ── Test Agentic Orchestrator ────────────────────────────────────────────────

class TestAgenticOrchestrator:
    def test_agentic_pipeline_local_only(self, phase4_cfg: PipelineConfig):
        from agent.orchestrator import AgenticRAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(
            chunks=[
                RetrievedChunk(
                    {
                        "text": "RAG reduces hallucinations",
                        "doc_id": "abcd1234",
                        "chunk_index": 0,
                        "source": "rag.txt",
                        "citation_id": "abcd1234:0",
                    },
                    score=0.95,
                )
            ],
            citations=["[abcd1234:0]"],
        )
        mock_retriever.enforcer = MagicMock()
        mock_retriever.enforcer.check_agentic.side_effect = lambda ans, chunks, web: MagicMock(
            is_valid=True, answer=ans
        )

        mock_mcp_manager = MagicMock()
        mock_mcp_manager.connect_all = AsyncMock()
        mock_mcp_manager.close_all = AsyncMock()
        mock_mcp_manager.get_tool_schemas.return_value = []

        pipeline = AgenticRAGPipeline(mock_retriever, mock_mcp_manager, phase4_cfg)

        # Mock LLM calls
        mock_llm = MagicMock()
        pipeline._llm = mock_llm

        # Step 1: Model calls local search
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "search_local_documents"
        tool_call.function.arguments = json.dumps({"query": "RAG"})

        msg1 = MagicMock()
        msg1.tool_calls = [tool_call]
        msg1.content = None

        # Step 2: Model returns final answer
        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = "RAG combines retrieval with generation [abcd1234:0]."

        resp1 = MagicMock(choices=[MagicMock(message=msg1)])
        resp2 = MagicMock(choices=[MagicMock(message=msg2)])
        mock_llm.chat.completions.create.side_effect = [resp1, resp2]

        ans = pipeline.answer("What is RAG?")
        assert "[abcd1234:0]" in ans.answer
        assert len(ans.tool_trace) == 1
        assert ans.tool_trace[0]["tool"] == "search_local_documents"


# ── Test MCP Server & Security ────────────────────────────────────────────────

class TestMCPServerSecurity:
    def test_path_traversal_rejection(self, tmp_path: Path):
        from mcp_server.server import _resolve_within_allowlist

        allowlist_root = tmp_path / "allowed"
        allowlist_root.mkdir()
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("secret content")

        with pytest.raises(ValueError) as exc_info:
            _resolve_within_allowlist(str(secret_file), str(allowlist_root))
        assert "Path traversal rejected" in str(exc_info.value)

    def test_valid_path_resolves(self, tmp_path: Path):
        from mcp_server.server import _resolve_within_allowlist

        allowlist_root = tmp_path / "allowed"
        allowlist_root.mkdir()
        valid_file = allowlist_root / "doc.txt"
        valid_file.write_text("valid content")

        resolved = _resolve_within_allowlist(str(valid_file), str(allowlist_root))
        assert resolved == valid_file.resolve()
