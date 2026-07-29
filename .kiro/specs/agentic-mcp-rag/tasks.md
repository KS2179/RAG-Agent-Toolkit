# Implementation Plan — Agentic Multi-Source RAG + MCP

Tasks are grouped into waves. Within a wave, tasks have no dependencies on each other and can run
concurrently. Each task lists which requirement(s) it satisfies so completion is checkable
against `requirements.md`.

## Wave 1 — Foundations (no dependencies on each other)

- [ ] 1. Add new dependencies to `requirements.txt`
  - Add `mcp`, `pypdf`, `python-docx`, `pylatexenc`, `filetype`
  - Do not modify or pin-bump any existing entry
  - _Requirements: 9 (design §9), 7.3_

- [ ] 2. Create `ingestion/parsers/base.py`
  - `ParsedDocument` dataclass (`text`, `source`, `metadata`)
  - `BaseParser` with `parse(path) -> ParsedDocument` and `can_parse(path, sniffed_type=None) -> bool`
  - `UnsupportedFormatError` exception
  - _Requirements: 1.7_

- [ ] 3. Implement `ingestion/parsers/text_parser.py`
  - Formalizes today's `read_text(encoding="utf-8", errors="replace")` behavior for `.txt`/`.md`
  - Unit test: extracted text matches file content exactly
  - _Requirements: 1.1_

- [ ] 4. Implement `ingestion/parsers/pdf_parser.py`
  - `pypdf.PdfReader`, concatenate per-page `extract_text()`, set `metadata["page_count"]`
  - Empty-text-layer PDFs return a `ParsedDocument` with empty text, not an exception
  - Unit tests: normal PDF extracts known text; scanned/no-text-layer PDF returns empty text
    without raising
  - _Requirements: 1.1, 1.4, 1.5_

- [ ] 5. Implement `ingestion/parsers/docx_parser.py`
  - `python-docx`, join paragraph text + table cell text in document order
  - Unit test: extracted text matches a known fixture `.docx`
  - _Requirements: 1.1, 1.5_

- [ ] 6. Implement `ingestion/parsers/latex_parser.py`
  - `pylatexenc.latex2text.LatexNodes2Text().latex_to_text(...)`
  - Config-driven `keep_math` flag (default `False`)
  - Unit test: a fixture `.tex` with commands/math/comments produces clean prose with no stray
    backslash commands or braces
  - _Requirements: 1.1, 1.6_

- [ ] 7. Implement `ingestion/parsers/registry.py`
  - `EXTENSION_TO_PARSER` map + `get_parser_for(path) -> BaseParser`
  - Extension lookup first, then `filetype` magic-byte sniff fallback, then
    `UnsupportedFormatError`
  - Unit tests: correct parser returned per extension; wrong extension resolved via sniffing;
    truly unsupported file raises the documented error
  - _Requirements: 1.3, 1.7_

- [ ] 8. Extend `retrieval/citation_enforcer.py` to accept mixed citation patterns
  - Add `WEB_CITATION = r"\[web:(\d+)\]"` alongside the existing local pattern
  - Validation accepts either form; a `[web:n]` is only valid if index `n` is in the supplied
    list of retrieved web sources for that answer
  - Existing local-only validation path (Phase 1-3) unchanged for callers that don't pass a web
    source list
  - Unit tests: pure-local answer still validates as before (regression); mixed-citation answer
    validates; invented `[web:n]` with no matching source triggers the existing retry path
  - _Requirements: 2.5, 7.2_

## Wave 2 — Ingestion pipeline integration (depends on Wave 1, tasks 2-7)

- [ ] 9. Wire `ingestion/pipeline.py`'s `ingest_file` through the parser registry
  - Replace direct `read_text` with `get_parser_for(path).parse(path)`, merge parsed metadata
    with caller-supplied metadata, then call existing `ingest_documents([...])` unchanged
  - _Requirements: 1.1_

- [ ] 10. Wire `ingest_directory` through the parser registry with a widened default glob
  - Default glob becomes `"**/*"`; non-file paths skipped; unsupported formats collected into a
    new `skipped: list[dict]` field on `IngestionStats` (filename + reason) instead of aborting
    the batch
  - Unit test: a directory with one supported and one unsupported file ingests the supported one
    and reports the other in `skipped`
  - _Requirements: 1.2, 1.3_

- [ ] 11. Add `config/phase4.yaml` with the new `ingestion.parsers` section
  - `enabled_formats`, `pdf.ocr_fallback` (default `false`), `latex.keep_math` (default `false`)
  - Extend `config/loader.py`'s Pydantic models with these new optional sections, defaulting such
    that `phase1.yaml`/`phase2.yaml`/`phase3.yaml` parse unchanged (no new required fields)
  - _Requirements: 5.1, 7.1_

## Wave 3 — Custom RAG MCP server (depends on Wave 2)

- [ ] 12. Scaffold `mcp_server/server.py` using `FastMCP`
  - Build pipeline/bundle once at module load via existing `build_ingestion_pipeline(config_path)`
  - _Requirements: 4.1_

- [ ] 13. Implement the `rag_query` tool
  - Delegates to existing `RAGPipeline.answer` / `HybridRAGPipeline`; returns answer text,
    citations, and chunk dicts (`Chunk.to_dict()`)
  - _Requirements: 4.1, 4.3_

- [ ] 14. Implement the `ingest_document` tool with allow-listed path resolution
  - `_resolve_within_allowlist(path)` against `mcp_server.ingest_allowlist_root`; reject anything
    that resolves outside it (including via symlink)
  - Delegates to existing `IngestionPipeline.ingest_file`
  - Unit test: traversal attempt (`../../etc/passwd`-style) is rejected; a legitimate path inside
    the allowlist root succeeds
  - _Requirements: 4.1, 4.2, 4.4_

- [ ] 15. Implement the `list_ingested_sources` tool
  - Reads distinct `source`/metadata values back from the vector store
  - _Requirements: 4.1_

- [ ] 16. Add `mcp_server/__main__.py` (`python -m mcp_server` entrypoint) and the
      `mcp.server` config section (transport, http_port, ingest_allowlist_root)
  - Defaults to stdio transport; `streamable-http` selectable via config
  - _Requirements: 4.5, 5.1_

- [ ] 17. Integration test: start the server in-process, connect a real MCP client session, call
      all three tools end-to-end against a temp Chroma directory
  - Include the path-traversal rejection test from task 14 exercised over an actual MCP call, not
    just the helper function directly
  - _Requirements: 4.1-4.5_

## Wave 4 — Agent orchestrator with local tool only (depends on Wave 1, task 8; independent of Wave 3)

- [ ] 18. Define `LOCAL_SEARCH_TOOL_SCHEMA` and `run_local_search()` in `agent/tools.py`
  - Tool result includes `chunks`, `top_score`, and `below_confidence_threshold`
  - _Requirements: 2.1, 2.3_

- [ ] 19. Implement `agent/orchestrator.py`'s `AgenticRAGPipeline` with local-tool-only support
  - Tool-calling loop against the configured LLM; bounded by `max_tool_iterations`
  - Builds a `tool_trace` of every tool call made, its arguments, and a result summary
  - Runs the extended citation enforcer (task 8) on the final answer
  - _Requirements: 2.1, 2.2, 2.4, 2.6, 2.7, 2.8_

- [ ] 20. Add `agent_system` prompt to `config/phase4.yaml`'s `prompts` section
  - Encodes the local-first / fallback-on-low-confidence / cite-by-source policy from design §4.4
  - _Requirements: 2.8, 5.1_

- [ ] 21. Unit tests for the orchestrator, all tools mocked
  - High-confidence local-only path makes no further calls
  - Empty/low-confidence local result is visible to the model via `below_confidence_threshold`
  - Max-iteration cutoff returns a best-effort answer rather than looping
  - "Nothing supports this claim" path returns an explicit don't-know rather than a fabricated
    answer
  - _Requirements: 2.2, 2.3, 2.4, 2.6_

## Wave 5 — MCP client + external tool routing (depends on Wave 4)

- [ ] 22. Implement `agent/mcp_client.py`'s `MCPClientManager`
  - `connect_all()` opens a session per enabled configured server (stdio or streamable-HTTP)
  - `get_tool_schemas()` returns every discovered tool, namespaced `f"{server}__{tool}"`
  - `call_tool(namespaced_name, arguments)` splits back to `(server, tool)`, wraps failures as a
    structured error result instead of raising
  - Unreachable/misconfigured server at `connect_all()` time logs a warning and is simply absent
    from `get_tool_schemas()` for that run, not a hard failure
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 23. Add `mcp.servers` config section to `config/phase4.yaml`
  - List of `{name, enabled, transport, command/url, args, env, tool_allowlist}`
  - `${VAR}` environment-variable substitution for secrets (e.g. `TAVILY_API_KEY`), never
    hardcoded into YAML
  - Document the recommended default (Tavily) and alternatives (Brave, Exa) per design §7
  - _Requirements: 3.1, 5.3_

- [ ] 24. Wire `MCPClientManager` into `AgenticRAGPipeline`
  - Tool list handed to the LLM becomes local tool + every namespaced MCP tool
  - Tool-call dispatch routes namespaced calls to `MCPClientManager.call_tool`, local calls to
    `run_local_search`
  - _Requirements: 2.1, 3.2, 3.3_

- [ ] 25. Implement the `strategy` config option (`auto` / `rag_only` / `rag_then_fallback`)
  - `rag_only`: never register external tool schemas
  - `rag_then_fallback`: call local search first unconditionally; only add external tool schemas
    to the model's next turn if `below_confidence_threshold` is true
  - `auto`: all tools available from the first turn (current default behavior from Wave 4/5)
  - _Requirements: 5.2_

- [ ] 26. Guard against local content leaking into external tool-call arguments
  - Add a test asserting the arguments passed to an external tool are derived from the user's
    question / model's reformulation, not raw local chunk text, for a scenario where both tools
    are called
  - _Requirements: 3.5_

- [ ] 27. Unit tests for `MCPClientManager` against a mock/in-memory MCP server
  - Tool namespacing, simulated timeout wrapped as an error result (not raised), a configured but
    unreachable server is absent from the schema list without crashing startup
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 28. End-to-end test: question requiring both local and external tools
  - Assert both tools appear in `tool_trace`, final answer contains both `[doc_id:chunk_index]`
    and `[web:n]` citations, and each citation type is valid per task 8's rules
  - _Requirements: 2.2, 2.5, 2.7_

## Wave 6 — Evaluation and observability (depends on Wave 5)

- [ ] 29. Extend `eval/dataset.jsonl` schema with an optional `expected_source` field
  - Values: `local`, `web`, `either`, `both`; existing unlabeled examples remain valid as-is
  - _Requirements: 6.1, 6.3_

- [ ] 30. Add a tool-selection metric to `eval/scorer.py` / `eval/run_eval.py`
  - For labeled examples only, compare `tool_trace` (which tool(s) were actually called) against
    `expected_source`; report alongside the existing four Phase 3 metrics, not replacing them
  - Unlabeled examples are excluded from this metric and unaffected otherwise
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 31. Update the GitHub Actions quality gate to include the new metric as informational (not
      blocking) until enough labeled examples exist to set a meaningful threshold
  - _Requirements: 6.2_

## Wave 7 — CLI wiring, docs, and final regression (depends on all prior waves)

- [ ] 32. Add an `--agentic` flag to `main.py`
  - When set, builds `AgenticRAGPipeline` (with `MCPClientManager` connected per config) instead
    of the plain `RAGPipeline`; default (`--agentic` absent) behavior is byte-for-byte unchanged
    from today
  - _Requirements: 7.1_

- [ ] 33. Add `python -m mcp_server` usage and `--agentic` usage to `README.md`
  - Include the recommended `mcp.servers` starter config (Tavily) and where to get an API key
  - _Requirements: none functional — documentation_

- [ ] 34. Run the full existing test suite (`test_phase1.py`, `test_phase2.py`, `test_phase3.py`)
      unmodified and confirm all 47 tests still pass
  - _Requirements: 7.2_

- [ ] 35. Confirm graceful degradation: with `mcp`, `pypdf`, `python-docx`, and `pylatexenc`
      uninstalled, plain `.txt`/`.md` ingestion and non-agentic `main.py` usage still work exactly
      as before
  - _Requirements: 7.3_
