"""
eval/build_dataset.py
────────────────────────────────────────────────────────────────────────────────
Curates and validates the evaluation dataset.

Two modes:
  1. SEED  — writes the hand-crafted baseline dataset (60 QA pairs) to
             eval/dataset.jsonl.  Run once to bootstrap.
  2. GENERATE — uses the LLM to auto-generate additional QA pairs from
                your ingested documents, then appends them.

Dataset schema (one JSON object per line):
{
  "id":           "q001",          # stable unique ID
  "question":     "...",           # natural-language question
  "ground_truth": "...",           # reference answer (used for context recall)
  "source_docs":  ["doc_a.txt"],   # which documents should be retrieved
  "difficulty":   "easy|medium|hard",
  "category":     "factual|reasoning|multi-hop|edge-case"
}

Usage:
  python eval/build_dataset.py --mode seed
  python eval/build_dataset.py --mode generate --n 40 --config config/phase3.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List



def validate_record(record: dict, idx: int) -> List[str]:
    """Return a list of validation errors for a single record."""
    errors = []
    required = {"id", "question", "ground_truth", "source_docs", "difficulty", "category"}
    for field in required:
        if field not in record:
            errors.append(f"Record {idx}: missing field '{field}'")

    if "difficulty" in record and record["difficulty"] not in {"easy", "medium", "hard"}:
        errors.append(f"Record {idx}: invalid difficulty '{record['difficulty']}'")

    if "category" in record and record["category"] not in {
        "factual", "reasoning", "multi-hop", "edge-case"
    }:
        errors.append(f"Record {idx}: invalid category '{record['category']}'")

    if "question" in record and len(record["question"].strip()) < 10:
        errors.append(f"Record {idx}: question too short")

    if "ground_truth" in record and len(record["ground_truth"].strip()) < 5:
        errors.append(f"Record {idx}: ground_truth too short")

    return errors



SEED_DATASET = [
    {"id": "q001", "question": "What is artificial intelligence?",
     "ground_truth": "Artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q002", "question": "How is AI research defined?",
     "ground_truth": "AI research is defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q003", "question": "How does AI differ from natural intelligence?",
     "ground_truth": "AI is intelligence demonstrated by machines, whereas natural intelligence is displayed by animals including humans.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "reasoning"},

    {"id": "q004", "question": "What does an intelligent agent do according to AI research?",
     "ground_truth": "An intelligent agent perceives its environment and takes actions that maximize its chance of achieving its goals.",
     "source_docs": ["ai_overview.txt"], "difficulty": "medium", "category": "factual"},

    {"id": "q005", "question": "Is AI the same as human intelligence?",
     "ground_truth": "No, AI is intelligence demonstrated by machines, which is distinct from the natural intelligence displayed by humans and other animals.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "reasoning"},

    {"id": "q006", "question": "What is the goal of an intelligent agent?",
     "ground_truth": "An intelligent agent takes actions that maximize its chance of achieving its goals.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q007", "question": "Can machines demonstrate intelligence?",
     "ground_truth": "Yes, artificial intelligence is defined as intelligence demonstrated by machines.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q008", "question": "What does an AI system perceive?",
     "ground_truth": "An AI system (intelligent agent) perceives its environment.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q009", "question": "Is AI research focused on environments or goals?",
     "ground_truth": "AI research focuses on intelligent agents that perceive their environment and take actions to maximize achievement of their goals — both aspects are central.",
     "source_docs": ["ai_overview.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q010", "question": "What animals are contrasted with machines in the definition of AI?",
     "ground_truth": "Animals including humans are contrasted with machines in the definition of AI.",
     "source_docs": ["ai_overview.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q011", "question": "What is Retrieval-Augmented Generation?",
     "ground_truth": "RAG is a technique that combines information retrieval with language model generation, first retrieving relevant documents from an external knowledge base and conditioning the generation on those documents.",
     "source_docs": ["rag_explained.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q012", "question": "How does RAG reduce hallucination?",
     "ground_truth": "RAG reduces hallucination by conditioning generation on retrieved documents rather than relying solely on parametric knowledge stored in model weights, keeping answers grounded.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q013", "question": "What does RAG retrieve documents from?",
     "ground_truth": "RAG retrieves documents from an external knowledge base.",
     "source_docs": ["rag_explained.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q014", "question": "What type of search does RAG typically use?",
     "ground_truth": "RAG typically uses dense vector search (semantic search) over embeddings stored in a vector database.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "factual"},

    {"id": "q015", "question": "What is parametric knowledge in the context of RAG?",
     "ground_truth": "Parametric knowledge refers to information stored in the model weights of a language model, as opposed to external knowledge retrieved at inference time.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "factual"},

    {"id": "q016", "question": "Name two vector databases mentioned in the context of RAG.",
     "ground_truth": "Chroma and Weaviate are mentioned as vector databases used in RAG systems.",
     "source_docs": ["rag_explained.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q017", "question": "Why is grounding important in RAG?",
     "ground_truth": "Grounding is important because it reduces hallucination by ensuring answers are based on retrieved documents rather than the model's internal parametric knowledge.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q018", "question": "Does RAG rely solely on model weights for answers?",
     "ground_truth": "No, RAG combines retrieval from an external knowledge base with generation, rather than relying solely on parametric knowledge stored in model weights.",
     "source_docs": ["rag_explained.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q019", "question": "What happens during the retrieval step in RAG?",
     "ground_truth": "During the retrieval step, relevant documents are retrieved from an external knowledge base using dense vector search over embeddings stored in a vector database.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "factual"},

    {"id": "q020", "question": "How does RAG condition the generation step?",
     "ground_truth": "RAG conditions generation on the retrieved documents, meaning the language model uses those documents as context when producing its answer.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q021", "question": "What are vector databases purpose-built for?",
     "ground_truth": "Vector databases are purpose-built systems for storing and querying high-dimensional embedding vectors.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q022", "question": "What ANN algorithm is commonly used in vector databases?",
     "ground_truth": "HNSW (Hierarchical Navigable Small World) is a commonly used ANN algorithm in vector databases.",
     "source_docs": ["vector_databases.txt"], "difficulty": "medium", "category": "factual"},

    {"id": "q023", "question": "Name three popular vector databases.",
     "ground_truth": "Popular vector databases include Chroma, Weaviate, Pinecone, Qdrant, and Milvus.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q024", "question": "Why is ChromaDB suitable for local development?",
     "ground_truth": "ChromaDB is suitable for local development because it requires no external infrastructure and persists data to disk automatically.",
     "source_docs": ["vector_databases.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q025", "question": "What does ANN stand for in the context of vector databases?",
     "ground_truth": "ANN stands for approximate nearest-neighbour.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q026", "question": "What makes HNSW fast at scale?",
     "ground_truth": "HNSW is an approximate nearest-neighbour search algorithm designed to make similarity search fast at scale.",
     "source_docs": ["vector_databases.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q027", "question": "Do vector databases require high-dimensional vectors?",
     "ground_truth": "Yes, vector databases are purpose-built for storing and querying high-dimensional embedding vectors.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q028", "question": "Does ChromaDB need external infrastructure?",
     "ground_truth": "No, ChromaDB requires no external infrastructure.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q029", "question": "What is Milvus?",
     "ground_truth": "Milvus is a popular vector database for storing and querying high-dimensional embedding vectors.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q030", "question": "How does ChromaDB persist data?",
     "ground_truth": "ChromaDB persists data to disk automatically.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "factual"},

    {"id": "q031", "question": "Which vector databases are specifically mentioned in the context of RAG systems?",
     "ground_truth": "Chroma and Weaviate are specifically mentioned as vector databases used in RAG systems.",
     "source_docs": ["rag_explained.txt", "vector_databases.txt"], "difficulty": "medium", "category": "multi-hop"},

    {"id": "q032", "question": "Could ChromaDB be used as the vector store in a RAG system?",
     "ground_truth": "Yes, ChromaDB is mentioned as a vector database suitable for RAG systems and is particularly good for local development.",
     "source_docs": ["rag_explained.txt", "vector_databases.txt"], "difficulty": "medium", "category": "multi-hop"},

    {"id": "q033", "question": "How do AI agents and RAG systems both relate to external environments?",
     "ground_truth": "AI agents perceive their environment to act; RAG systems retrieve from an external knowledge base to generate grounded answers — both rely on external information rather than purely internal state.",
     "source_docs": ["ai_overview.txt", "rag_explained.txt"], "difficulty": "hard", "category": "multi-hop"},

    {"id": "q034", "question": "What role do embeddings play in connecting RAG and vector databases?",
     "ground_truth": "RAG uses dense vector search over embeddings stored in a vector database during its retrieval step; vector databases are purpose-built to store and query these high-dimensional embedding vectors efficiently.",
     "source_docs": ["rag_explained.txt", "vector_databases.txt"], "difficulty": "hard", "category": "multi-hop"},

    {"id": "q035", "question": "Is Weaviate suitable for a RAG system's retrieval step?",
     "ground_truth": "Yes, Weaviate is a popular vector database mentioned in the context of RAG and is capable of storing embeddings and supporting similarity search.",
     "source_docs": ["rag_explained.txt", "vector_databases.txt"], "difficulty": "medium", "category": "multi-hop"},

    {"id": "q036", "question": "Why would you choose Chroma over Pinecone for a prototype?",
     "ground_truth": "ChromaDB requires no external infrastructure and persists data to disk automatically, making it ideal for local development and prototyping without setup overhead.",
     "source_docs": ["vector_databases.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q037", "question": "What is the advantage of conditioning generation on retrieved documents?",
     "ground_truth": "Conditioning on retrieved documents reduces hallucination and keeps answers grounded, rather than relying on potentially outdated or inaccurate parametric knowledge.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "reasoning"},

    {"id": "q038", "question": "Why is approximate nearest-neighbour search used instead of exact search?",
     "ground_truth": "ANN algorithms like HNSW make similarity search fast at scale — exact search over millions of high-dimensional vectors would be too slow.",
     "source_docs": ["vector_databases.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q039", "question": "How does RAG improve on a model that only uses its weights?",
     "ground_truth": "RAG retrieves up-to-date, relevant documents at inference time, reducing hallucination and allowing the model to answer questions beyond its training data, unlike a model that only uses parametric knowledge.",
     "source_docs": ["rag_explained.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q040", "question": "What trade-off does ANN search make compared to exact similarity search?",
     "ground_truth": "ANN search trades some accuracy (approximate rather than exact results) for significant speed gains, enabling fast similarity search at scale.",
     "source_docs": ["vector_databases.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q041", "question": "Is Qdrant mentioned as a vector database?",
     "ground_truth": "Yes, Qdrant is listed as one of the popular vector database options.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "edge-case"},

    {"id": "q042", "question": "Does the context say AI is better than human intelligence?",
     "ground_truth": "No, the context only defines AI as intelligence demonstrated by machines, contrasting it with natural intelligence — it makes no claim about superiority.",
     "source_docs": ["ai_overview.txt"], "difficulty": "medium", "category": "edge-case"},

    {"id": "q043", "question": "Does RAG eliminate hallucination entirely?",
     "ground_truth": "The context states RAG reduces hallucination and keeps answers grounded, but does not claim it eliminates hallucination entirely.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "edge-case"},

    {"id": "q044", "question": "Is HNSW the only ANN algorithm mentioned?",
     "ground_truth": "Yes, HNSW (Hierarchical Navigable Small World) is the only ANN algorithm explicitly mentioned in the context.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "edge-case"},

    {"id": "q045", "question": "What is the capital of France?",
     "ground_truth": "The context does not contain information about the capital of France.",
     "source_docs": [], "difficulty": "easy", "category": "edge-case"},

    {"id": "q046", "question": "Does the context mention GPT-4?",
     "ground_truth": "No, GPT-4 is not mentioned in the context.",
     "source_docs": [], "difficulty": "easy", "category": "edge-case"},

    {"id": "q047", "question": "Is LangChain mentioned as a RAG framework?",
     "ground_truth": "No, LangChain is not mentioned in the context. The context refers to Chroma and Weaviate as vector databases but does not name LangChain.",
     "source_docs": [], "difficulty": "easy", "category": "edge-case"},

    {"id": "q048", "question": "Does ChromaDB support GPU acceleration according to the context?",
     "ground_truth": "The context does not mention GPU acceleration for ChromaDB.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "edge-case"},

    {"id": "q049", "question": "What is the cost of using Pinecone?",
     "ground_truth": "The context does not provide information about Pinecone's pricing.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "edge-case"},

    {"id": "q050", "question": "Is Weaviate open source?",
     "ground_truth": "The context does not mention whether Weaviate is open source.",
     "source_docs": ["vector_databases.txt"], "difficulty": "easy", "category": "edge-case"},

    {"id": "q051", "question": "If a RAG system retrieves wrong documents, what happens to faithfulness?",
     "ground_truth": "If wrong documents are retrieved, the generated answer would be conditioned on irrelevant context, likely reducing faithfulness since claims would not be supported by the correct source material.",
     "source_docs": ["rag_explained.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q052", "question": "Why might you use multiple vector databases in production?",
     "ground_truth": "The context lists several options (Chroma, Weaviate, Pinecone, Qdrant, Milvus) with different trade-offs; Chroma suits local dev while others may offer better scalability or cloud integration for production.",
     "source_docs": ["vector_databases.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q053", "question": "What distinguishes an intelligent agent from a simple program?",
     "ground_truth": "An intelligent agent perceives its environment and takes actions that maximize its chance of achieving its goals, implying adaptive, goal-directed behaviour rather than fixed execution.",
     "source_docs": ["ai_overview.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q054", "question": "How does dense vector search differ from keyword search?",
     "ground_truth": "Dense vector search uses embeddings to find semantically similar content, whereas keyword search matches exact terms. RAG uses dense vector search, which can find relevant passages even when exact words differ.",
     "source_docs": ["rag_explained.txt", "vector_databases.txt"], "difficulty": "hard", "category": "multi-hop"},

    {"id": "q055", "question": "What problem does RAG solve that a standalone LLM cannot?",
     "ground_truth": "RAG solves the hallucination and knowledge-staleness problem by retrieving up-to-date external documents at inference time, rather than relying solely on parametric knowledge baked into model weights.",
     "source_docs": ["rag_explained.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q056", "question": "Why does vector search need specialised databases instead of relational ones?",
     "ground_truth": "Relational databases are not optimised for high-dimensional vector similarity search; vector databases use specialised ANN algorithms like HNSW to make such search fast at scale.",
     "source_docs": ["vector_databases.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q057", "question": "Could you build a RAG system without a vector database?",
     "ground_truth": "The context describes RAG as typically using dense vector search over embeddings stored in a vector database, implying the vector database is a standard component, though the context does not explicitly say alternatives are impossible.",
     "source_docs": ["rag_explained.txt", "vector_databases.txt"], "difficulty": "hard", "category": "reasoning"},

    {"id": "q058", "question": "How does the definition of an AI agent relate to RAG retrieval?",
     "ground_truth": "An AI agent perceives its environment and acts to achieve goals; a RAG system similarly perceives the query and retrieves relevant context — retrieval is the agent's perception step in grounded generation.",
     "source_docs": ["ai_overview.txt", "rag_explained.txt"], "difficulty": "hard", "category": "multi-hop"},

    {"id": "q059", "question": "What would happen if the vector database went offline in a RAG system?",
     "ground_truth": "The context does not describe failure scenarios, but since RAG retrieval relies on the vector database, an offline database would prevent retrieval and likely cause the system to fail or fall back to parametric knowledge.",
     "source_docs": ["rag_explained.txt"], "difficulty": "hard", "category": "edge-case"},

    {"id": "q060", "question": "Is there a difference between semantic search and dense vector search?",
     "ground_truth": "The context uses the terms interchangeably, referring to dense vector search as semantic search — both describe finding relevant content based on meaning via embeddings.",
     "source_docs": ["rag_explained.txt"], "difficulty": "medium", "category": "reasoning"},
]


def cmd_seed(output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_errors = []
    for i, record in enumerate(SEED_DATASET):
        errors = validate_record(record, i)
        all_errors.extend(errors)

    if all_errors:
        for e in all_errors:
            print(f"VALIDATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    with out.open("w") as f:
        for record in SEED_DATASET:
            f.write(json.dumps(record) + "\n")

    print(f"✓ Wrote {len(SEED_DATASET)} QA pairs to {out}")


def cmd_generate(output_path: str, n: int, config_path: str) -> None:
    """Auto-generate n additional QA pairs using the LLM."""
    import os, random
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out    = Path(output_path)

    existing = []
    if out.exists():
        with out.open() as f:
            existing = [json.loads(l) for l in f if l.strip()]

    next_id  = len(existing) + 1
    existing_questions = {r["question"].lower() for r in existing}

    GENERATION_PROMPT = """Generate {n} diverse evaluation QA pairs for a RAG system.
Each pair must be a JSON object on its own line with these fields:
- id: string like "q{num:03d}"
- question: a natural language question
- ground_truth: the correct answer
- source_docs: list of relevant doc names from: ["ai_overview.txt", "rag_explained.txt", "vector_databases.txt"]
- difficulty: one of "easy", "medium", "hard"
- category: one of "factual", "reasoning", "multi-hop", "edge-case"

Vary difficulty and category. Include some edge cases where the answer is not in the docs.
Return ONLY the JSON lines, no markdown, no explanation.
Start IDs from q{start:03d}."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": GENERATION_PROMPT.format(
            n=n, num=next_id, start=next_id
        )}],
        temperature=0.8,
    )

    new_records = []
    for line in response.choices[0].message.content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("question", "").lower() in existing_questions:
                continue  
            errors = validate_record(record, len(new_records))
            if not errors:
                new_records.append(record)
        except json.JSONDecodeError:
            continue

    with out.open("a") as f:
        for record in new_records:
            f.write(json.dumps(record) + "\n")

    print(f"✓ Appended {len(new_records)} generated QA pairs to {out}")
    print(f"  Total dataset size: {len(existing) + len(new_records)}")


def cmd_validate(dataset_path: str) -> None:
    path = Path(dataset_path)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    records = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR line {i+1}: invalid JSON — {e}", file=sys.stderr)
                sys.exit(1)

    ids = [r.get("id") for r in records]
    all_errors = []
    for i, record in enumerate(records):
        all_errors.extend(validate_record(record, i))

    seen = set()
    for id_ in ids:
        if id_ in seen:
            all_errors.append(f"Duplicate ID: {id_}")
        seen.add(id_)

    if all_errors:
        for e in all_errors:
            print(f"VALIDATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Dataset valid — {len(records)} records, {len(seen)} unique IDs")
    cats = {}
    for r in records:
        cats[r.get("category", "?")] = cats.get(r.get("category", "?"), 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "generate", "validate"],
                        default="seed")
    parser.add_argument("--output", default="eval/dataset.jsonl")
    parser.add_argument("--n", type=int, default=40,
                        help="Number of QA pairs to generate (generate mode)")
    parser.add_argument("--config", default="config/phase3.yaml")
    args = parser.parse_args()

    if args.mode == "seed":
        cmd_seed(args.output)
    elif args.mode == "generate":
        cmd_generate(args.output, args.n, args.config)
    elif args.mode == "validate":
        cmd_validate(args.output)
