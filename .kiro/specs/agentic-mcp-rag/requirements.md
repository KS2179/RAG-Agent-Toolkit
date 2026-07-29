# Requirements — Agentic Multi-Source RAG + MCP

## Context

`mcp-rag-pipeline` currently (Phases 1-3) does the following, and none of it should regress:

- Ingests **plain-text files only** (`ingest_file`/`ingest_directory` call `Path.read_text()`
  directly — no PDF, Word, or LaTeX support exists today).
- Chunks with a sliding-window token chunker, embeds via OpenAI or local sentence-transformers,
  stores in ChromaDB or Weaviate.
- Retrieves with either pure vector search (Phase 1) or BM25+vector hybrid with cross-encoder
  reranking (Phase 2).
- Answers by **always** retrieving then generating in one fixed path
  (`RAGPipeline.answer()` → `Retriever.retrieve()` → OpenAI chat completion). There is no
  decision point, no tool-calling, and no notion of "the model chose not to use RAG."
- Enforces inline citations of the form `[doc_id:chunk_index]` and evaluates faithfulness /
  relevance / context recall / citation coverage via an LLM-judge in Phase 3.
- Has **no MCP code at all** — neither client nor server — today.

This spec (Phase 4) adds: (1) ingestion for arbitrary document formats, (2) an agentic
orchestrator that lets the LLM decide whether to answer from local RAG, from an external tool
reached over MCP, from both, or to say it doesn't know, and (3) a custom MCP server that exposes
this project's own RAG pipeline as reusable, protocol-compliant tools.

## Requirement 1 — Multi-format document ingestion

**User story:** As a user of the pipeline, I want to ingest PDFs, Word documents, and LaTeX
source files (not just `.txt`), so that I can build a knowledge base out of the documents I
actually have instead of converting everything to plain text by hand first.

**Acceptance criteria:**

1.1. WHEN a file with extension `.pdf`, `.docx`, `.tex`, `.md`, or `.txt` is passed to
`ingest_file` or discovered by `ingest_directory`, THE SYSTEM SHALL extract its text content and
route it through the existing `ingest_documents(List[dict])` API unchanged.

1.2. WHEN `ingest_directory` is called without an explicit `glob`, THE SYSTEM SHALL discover
files of all supported extensions by default, rather than only `**/*.txt` as it does today.

1.3. IF a discovered file has an unsupported or undetectable format, THEN THE SYSTEM SHALL skip
that file, record it in the ingestion result as a warning (filename + reason), and continue
ingesting the remaining files rather than aborting the batch.

1.4. WHEN a PDF is a scanned image with no extractable text layer, THE SYSTEM SHALL report zero
extractable chunks for that file rather than silently producing empty or garbage chunks; OCR
fallback (if enabled via config) is a stretch goal, not a hard requirement of this phase.

1.5. WHEN a document is parsed, THE SYSTEM SHALL attach format-specific metadata (e.g., page
count for PDFs, page/section markers where available) to the resulting document dict's
`metadata` field so it can be surfaced in citations later.

1.6. WHERE a `.tex` file contains LaTeX markup (commands, math environments, comments), THE
SYSTEM SHALL convert it to readable prose text before chunking, rather than passing raw LaTeX
source (with backslash commands and braces) into the chunker.

1.7. THE SYSTEM SHALL treat parser selection as a pluggable registry keyed by file extension (and
a magic-byte fallback when the extension is missing or wrong), so a new format can be added by
registering one new parser class without touching `IngestionPipeline`.

## Requirement 2 — Agentic decision-making between local RAG and external tools

**User story:** As a user asking a question, I want the LLM to decide for itself whether the
answer should come from my ingested documents, from a live web search, or both, so that I get
grounded answers for things in my corpus and current answers for things that aren't.

**Acceptance criteria:**

2.1. WHEN a question is submitted in agentic mode, THE SYSTEM SHALL present the LLM with a set of
callable tools (at minimum: local document search, plus any tools discovered from connected MCP
servers) using standard tool-calling / function-calling, rather than unconditionally retrieving
and injecting local context as Phase 1-3 do.

2.2. THE SYSTEM SHALL allow the LLM to call zero, one, or multiple tools — including calling both
local search and an external tool for the same question — before producing a final answer.

2.3. WHEN local retrieval is used and its top result score falls below the configured
`score_threshold`, THE SYSTEM SHALL make that low-confidence signal available to the model (e.g.
via the tool result) so it can choose to fall back to an external tool rather than answering from
weak context.

2.4. THE SYSTEM SHALL enforce a maximum number of tool-call iterations (configurable, default 4)
per question, after which it returns the best available answer rather than looping indefinitely.

2.5. WHEN the final answer draws on local document chunks, THE SYSTEM SHALL cite them with the
existing `[doc_id:chunk_index]` format. WHEN the final answer draws on an external/web source, THE
SYSTEM SHALL cite it distinguishably (e.g. `[web:n]` with the source URL listed), so a reader can
tell local corpus claims apart from web claims at a glance.

2.6. IF the LLM cannot produce an answer supported by any tool result, THEN THE SYSTEM SHALL say
so explicitly rather than fabricating an answer.

2.7. THE SYSTEM SHALL log, per question, which tool(s) were called, in what order, with what
arguments, and a summary of what each returned — sufficient to answer "why did it choose that
source?" after the fact without re-running the query.

2.8. THE SYSTEM SHALL default to preferring local RAG when a question plausibly matches the
ingested corpus and no freshness signal is present, since it is cheaper, private (no external
network egress), and already citation-enforced — external tools are for cases local retrieval
can't cover, not a default first hop.

## Requirement 3 — MCP client: connecting to external MCP servers

**User story:** As the operator of this pipeline, I want to connect the agent to one or more
external MCP servers (e.g. a web search provider), and configure/swap which ones are active,
without changing the orchestrator's code.

**Acceptance criteria:**

3.1. THE SYSTEM SHALL read a list of MCP servers to connect to from configuration (name,
transport, connection details, enabled flag), and connect to each enabled server at startup.

3.2. WHEN connected to an MCP server, THE SYSTEM SHALL discover that server's available tools
and expose them to the orchestrator's tool-calling loop, namespaced by server name so tools from
different servers never collide (e.g. `tavily__web_search` vs. `internal_rag__rag_query`).

3.3. IF more than one MCP server is connected and multiple could plausibly answer a query (e.g.
two different search providers), THEN THE SYSTEM SHALL let the LLM choose which specific tool to
call based on the tool descriptions provided, rather than the orchestrator hard-coding which
server wins.

3.4. IF an MCP server is unreachable or a tool call to it times out, THEN THE SYSTEM SHALL
surface that failure to the LLM as a tool error (not a crash), so it can retry, try a different
tool, or fall back to answering from whatever it already has.

3.5. THE SYSTEM SHALL NOT send the full text of local documents to an external MCP server as part
of a tool call unless the query itself (as written by the user or reformulated by the model)
requires it — local corpus content is not leaked into web search queries by default.

## Requirement 4 — Custom MCP server exposing this project's RAG pipeline

**User story:** As a user of MCP-compatible clients in general (this project's own agent, Claude
Desktop, Cursor, Kiro, etc.), I want this pipeline's ingestion and retrieval to be available as a
standard MCP server, so the RAG capability isn't locked to one calling application.

**Acceptance criteria:**

4.1. THE SYSTEM SHALL expose an MCP server (stdio transport by default) with at minimum these
tools: query the RAG pipeline for an answer, ingest a document, and list currently ingested
sources.

4.2. WHEN the MCP server's ingest tool is invoked, THE SYSTEM SHALL reuse the existing
`IngestionPipeline`/`build_ingestion_pipeline` code path rather than duplicating chunking/
embedding/storage logic.

4.3. WHEN the MCP server's query tool is invoked, THE SYSTEM SHALL reuse the existing
`Retriever`/`HybridRetriever` and `RAGPipeline`/`HybridRAGPipeline` classes rather than
duplicating retrieval or generation logic.

4.4. IF the ingest tool receives a file path, THEN THE SYSTEM SHALL restrict reads to an
allow-listed root directory (configurable) and reject path-traversal attempts (e.g. `../../etc`),
since this server may be reachable by remote MCP clients and must not become an arbitrary file
read.

4.5. THE SYSTEM SHALL support this server running standalone (so external MCP clients can use it
without this project's own agent running), and being used as one of the tool sources by this
project's own orchestrator (Requirement 2/3) — the same server, two ways of reaching it.

## Requirement 5 — Configuration and extensibility

**User story:** As the operator, I want the new agentic/MCP/parser behavior to be config-driven
like Phases 1-3 already are, so I can turn features on/off and swap providers without touching
code.

**Acceptance criteria:**

5.1. THE SYSTEM SHALL add a versioned `config/phase4.yaml` that extends (not replaces) the
existing config schema — Phase 1-3 configs and code SHALL continue to work unmodified.

5.2. THE SYSTEM SHALL make the agent's routing strategy configurable among at least: `auto` (LLM
decides), `rag_only` (disable external tools), and `rag_then_fallback` (try local first, only
expose external tools to the model if local confidence is below threshold).

5.3. THE SYSTEM SHALL make the list of connected MCP servers, the local-confidence threshold, and
the max tool-call iterations all configurable without code changes.

## Requirement 6 — Observability and evaluation of routing decisions

**User story:** As the operator, I want to know whether the agent is choosing the right source
for a given question, not just whether its final answer is faithful, so I can catch cases where
it over- or under-uses web search.

**Acceptance criteria:**

6.1. THE SYSTEM SHALL extend the Phase 3 evaluation dataset format to optionally label each QA
pair with an expected source (`local`, `web`, `either`, `both`).

6.2. WHEN running evaluation against labeled examples, THE SYSTEM SHALL report a "tool selection"
metric alongside the existing faithfulness / relevance / context recall / citation coverage
metrics, comparing the tool(s) actually called against the expected source label.

6.3. THE SYSTEM SHALL NOT require every existing (unlabeled) Phase 3 eval example to be re-labeled
— examples without a source label are simply excluded from the tool-selection metric, not from
the rest of the eval suite.

## Requirement 7 — Backward compatibility

**User story:** As a maintainer, I don't want Phase 4 to break the working Phase 1-3 system.

**Acceptance criteria:**

7.1. THE SYSTEM SHALL keep `main.py`'s existing `--retrieval-only` / `--config` behavior working
unchanged; agentic mode SHALL be opt-in via a new flag, not the new default entry point.

7.2. THE SYSTEM SHALL keep all 47 existing tests (`tests/test_phase1.py`, `test_phase2.py`,
`test_phase3.py`) passing without modification.

7.3. IF the new ingestion parsers or agent/MCP dependencies are not installed, THEN THE SYSTEM
SHALL continue to support plain `.txt`/`.md` ingestion and non-agentic Phase 1-3 usage exactly as
it does today — the new features degrade gracefully rather than becoming hard requirements of
the whole package.
