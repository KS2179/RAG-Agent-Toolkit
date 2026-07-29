"""
eval/run_eval.py
────────────────────────────────────────────────────────────────────────────────
Offline evaluation script.  Wired to CI via .github/workflows/eval.yml.

Flow:
  1. Load eval dataset from eval/dataset.jsonl
  2. For each QA pair, run the RAG pipeline to get an answer + context
  3. Score each answer with FaithfulnessScorer (faithfulness, relevance,
     recall, citation coverage)
  4. Write per-sample results to eval/results.jsonl
  5. Write aggregate report to eval/report.json
  6. Exit 0 if all thresholds pass, Exit 1 if any breach (CI gate)

Usage:
  python eval/run_eval.py --config config/phase3.yaml
  python eval/run_eval.py --config config/phase3.yaml --sample 20
  python eval/run_eval.py --config config/phase3.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# Add project root to path so imports resolve regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import load_config
from eval.scorer import FaithfulnessScorer, SampleScore, build_report
from ingestion.pipeline import build_ingestion_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_rag_for_sample(
    sample: dict,
    retriever,
) -> tuple[str, str]:
    """
    Run the retrieval pipeline for a single QA pair.
    Returns (answer_text, context_block).

    In dry-run mode the retriever is None — returns placeholder values.
    """
    if retriever is None:
        return "[DRY RUN ANSWER] The answer is here [abcd1234:0].", "No context."

    result = retriever.retrieve(sample["question"])
    # In retrieval-only mode the "answer" is a placeholder; the scorer uses
    # the context_block to evaluate context recall and citation coverage.
    answer_placeholder = (
        f"[Retrieval-only] Retrieved {len(result.chunks)} chunk(s) "
        f"for: {sample['question']}"
    )
    return answer_placeholder, result.context_block


def run_full_rag(sample: dict, rag) -> tuple[str, str]:
    """Run retrieve + generate. Returns (answer, context)."""
    if rag is None:
        return "[DRY RUN ANSWER] Placeholder [abcd1234:0].", "No context."

    answer_obj = rag.answer(sample["question"])
    return answer_obj.answer, answer_obj.context_block


# ── Main eval loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Script — Phase 3")
    parser.add_argument("--config",   default="config/phase3.yaml")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Random sample size (default: full dataset)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Skip RAG + LLM calls; use placeholder answers")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Score retrieved context without LLM generation")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    eval_cfg = cfg.evaluation

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    dataset_path = eval_cfg.dataset_path
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run:  python eval/build_dataset.py --mode seed")
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} QA pairs from {dataset_path}")

    # ── 2. Enforce minimum dataset size ───────────────────────────────────────
    min_size = eval_cfg.ci.min_dataset_size
    if len(dataset) < min_size:
        print(f"ERROR: Dataset has {len(dataset)} records; minimum is {min_size}")
        sys.exit(1)

    # ── 3. Optional random sample ─────────────────────────────────────────────
    sample_size = args.sample or eval_cfg.ci.sample_size
    if sample_size and sample_size < len(dataset):
        dataset = random.sample(dataset, sample_size)
        print(f"Sampled {sample_size} records for evaluation")

    # ── 4. Build pipeline ─────────────────────────────────────────────────────
    if args.dry_run:
        print("DRY RUN — skipping RAG pipeline initialisation")
        retriever = None
        rag       = None
    else:
        print(f"Building pipeline from {args.config}...")
        ingestion, bundle = build_ingestion_pipeline(args.config)

        # Ingest sample docs if store is empty
        if bundle.retriever.vector_store.count() == 0:
            print("Vector store is empty — ingesting sample documents...")
            from main import SAMPLE_DOCS
            stats = ingestion.ingest_documents(SAMPLE_DOCS)
            print(f"  {stats}")

        retriever = bundle.retriever
        rag       = bundle.rag if not args.retrieval_only else None

    # ── 5. Run RAG for each sample ────────────────────────────────────────────
    print(f"\nRunning RAG pipeline on {len(dataset)} samples...")
    scored_inputs = []

    for i, sample in enumerate(dataset):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dataset)}...")

        try:
            if args.retrieval_only or rag is None:
                context, _ = run_rag_for_sample(sample, retriever)
                answer = f"[Retrieval-only mode] Context retrieved for: {sample['question']}"
            else:
                answer, context = run_full_rag(sample, rag)
        except Exception as e:
            answer  = ""
            context = f"ERROR: {e}"

        scored_inputs.append({
            "id":           sample["id"],
            "question":     sample["question"],
            "answer":       answer,
            "context":      context,
            "ground_truth": sample["ground_truth"],
        })

    # ── 6. Score ──────────────────────────────────────────────────────────────
    print(f"\nScoring {len(scored_inputs)} answers...")
    scorer  = FaithfulnessScorer(cfg)
    scores: List[SampleScore] = scorer.score_batch(
        scored_inputs,
        max_workers=eval_cfg.scorer.max_workers,
    )

    # ── 7. Write per-sample results ───────────────────────────────────────────
    results_path = Path(eval_cfg.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w") as f:
        for score in scores:
            f.write(json.dumps(asdict(score)) + "\n")

    print(f"Results written to {results_path}")

    # ── 8. Build and write aggregate report ───────────────────────────────────
    report = build_report(scores, dict(eval_cfg.thresholds))
    report.print_summary()

    report_path = Path(eval_cfg.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Report written to {report_path}")

    # ── 9. CI gate ────────────────────────────────────────────────────────────
    if eval_cfg.ci.fail_on_threshold_breach and not report.ci_passed:
        print(f"\n❌ CI FAILED — thresholds breached: {report.threshold_breaches}")
        sys.exit(1)

    print("✅ CI PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
