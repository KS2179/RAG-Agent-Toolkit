# RAG Pipeline

A modular Retrieval-Augmented Generation system with hybrid search, citation enforcement, and automated faithfulness evaluation. Built as the foundation for an MCP-powered agentic RAG system.

## Quick Start

```bash
git clone https://github.com/Vishisht3/mcp-rag-pipeline.git
cd mcp-rag-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env

python main.py                   # full RAG (requires API key)
python main.py --retrieval-only  # retrieval only, no API key needed
```

## Architecture

```
rag_pipeline/
├── config/
│   ├── loader.py              # Typed Pydantic config model
│   ├── phase1.yaml            # Phase 1: core RAG
│   ├── phase2.yaml            # Phase 2: hybrid search + reranking
│   └── phase3.yaml            # Phase 3: evaluation + CI
├── ingestion/
│   ├── chunker.py             # Sliding-window token chunker
│   ├── embedder.py            # OpenAI / local sentence-transformers
│   └── pipeline.py            # Orchestrates chunk -> embed -> store
├── store/
│   └── vector_store.py        # ChromaDB / Weaviate backend
├── retrieval/
│   ├── retriever.py           # Semantic top-k retrieval + citations
│   ├── hybrid_retriever.py    # BM25 + vector score fusion
│   ├── reranker.py            # Cross-encoder reranking
│   ├── bm25_index.py          # BM25 keyword index
│   └── citation_enforcer.py   # Post-generation citation validation
├── eval/
│   ├── build_dataset.py       # QA evaluation dataset management
│   ├── run_eval.py            # Evaluation runner
│   ├── scorer.py              # LLM-as-judge faithfulness scorer
│   └── dataset.jsonl          # Evaluation QA pairs
├── tests/
│   ├── test_phase1.py         # Chunker, pipeline, retriever tests
│   ├── test_phase2.py         # Hybrid search, reranker, citation tests
│   └── test_phase3.py         # Scorer and evaluation pipeline tests
├── main.py                    # End-to-end demo
└── requirements.txt
```

## Phases

Each phase is fully driven by its own YAML config — no code changes needed to switch between them.

### Phase 1 — Core RAG

```bash
python main.py --config config/phase1.yaml
```

- Sliding-window chunking (500-800 tokens, 100-token overlap)
- OpenAI `text-embedding-3-small` embeddings
- ChromaDB vector store with cosine similarity search
- Inline citation formatting: `[doc_id:chunk_index]`

### Phase 2 — Hybrid Search and Reranking

```bash
python main.py --config config/phase2.yaml
```

- Hybrid retrieval: BM25 keyword scores fused with vector similarity (alpha = 0.6/0.4)
- Cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Citation enforcement: validates inline citations post-generation, retries on violation

### Phase 3 — Evaluation and CI

```bash
python eval/run_eval.py --config config/phase3.yaml
```

- LLM-as-judge scoring via GPT-4o-mini across four metrics
- Configurable pass/fail thresholds
- GitHub Actions quality gate that blocks PRs on metric regression

## Evaluation Metrics

| Metric | What it measures | Default threshold |
|--------|-----------------|-------------------|
| Faithfulness | Are all answer claims supported by the context? | 0.75 |
| Answer Relevance | Does the answer address the question? | 0.70 |
| Context Recall | Does retrieved context cover the ground truth? | 0.65 |
| Citation Coverage | Does the answer contain valid inline citations? | 0.80 |

```bash
# Seed the evaluation dataset
python eval/build_dataset.py --mode seed --output eval/dataset.jsonl

# Full evaluation (requires OPENAI_API_KEY)
python eval/run_eval.py --config config/phase3.yaml

# Retrieval-only evaluation (no LLM cost)
python eval/run_eval.py --config config/phase3.yaml --retrieval-only
```

## Switching Vector Stores

Change one line in any config YAML to swap from ChromaDB to Weaviate:

```yaml
vector_store:
  backend: "weaviate"
  weaviate:
    url: "http://localhost:8080"
```

No code changes required.

## Running Tests

```bash
pytest tests/ -v
```

47 tests pass. 7 are skipped when `rank-bm25` or `sentence-transformers` are not installed — install them from `requirements.txt` to run the full suite.

## Tech Stack

- Python 3.11+
- OpenAI — embeddings and generation
- ChromaDB / Weaviate — vector storage
- sentence-transformers — local embeddings and cross-encoder reranking
- rank-bm25 — keyword search
- Pydantic — typed configuration
- GitHub Actions — CI evaluation pipeline
