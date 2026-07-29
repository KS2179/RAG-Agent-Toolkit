"""
eval/scorer.py
LLM-as-judge faithfulness scorer.

Measures four metrics per QA pair:
  1. faithfulness       — are all answer claims supported by the context?
  2. answer_relevance   — does the answer address the question?
  3. context_recall     — does the context cover the ground truth?
  4. citation_coverage  — does the answer contain valid inline citations?

All judge prompts live in config/phase3.yaml — never hardcoded here.
Uses concurrent.futures for batched parallel scoring.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from config.loader import PipelineConfig


# ── Per-sample result ─────────────────────────────────────────────────────────

@dataclass
class SampleScore:
    id: str
    question: str
    answer: str
    ground_truth: str
    faithfulness: float         = 0.0
    answer_relevance: float     = 0.0
    context_recall: float       = 0.0
    citation_coverage: float    = 0.0
    tool_selection_accuracy: Optional[float] = None
    expected_source: Optional[str] = None
    actual_tools: list          = field(default_factory=list)
    faithfulness_claims: list   = field(default_factory=list)
    faithfulness_reasoning: str = ""
    relevance_reasoning: str    = ""
    recall_reasoning: str       = ""
    error: Optional[str]        = None

    def passed(self, thresholds: dict) -> bool:
        pass_ts = True
        if self.tool_selection_accuracy is not None and "tool_selection" in thresholds:
            pass_ts = self.tool_selection_accuracy >= thresholds["tool_selection"]

        return (
            self.faithfulness     >= thresholds.get("faithfulness", 0.0)
            and self.answer_relevance  >= thresholds.get("answer_relevance", 0.0)
            and self.context_recall    >= thresholds.get("context_recall", 0.0)
            and self.citation_coverage >= thresholds.get("citation_coverage", 0.0)
            and pass_ts
        )


# ── Aggregate report ──────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_recall: float
    avg_citation_coverage: float
    avg_tool_selection: Optional[float]
    thresholds: dict
    threshold_breaches: List[str]   # metric names that fell below threshold
    samples: List[SampleScore]

    @property
    def ci_passed(self) -> bool:
        return len(self.threshold_breaches) == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci_passed"] = self.ci_passed
        d.pop("samples")    # stored separately in results.jsonl
        return d

    def print_summary(self) -> None:
        status = "✅ PASSED" if self.ci_passed else "❌ FAILED"
        print(f"\n{'─'*60}")
        print(f"Eval Report  {status}")
        print(f"{'─'*60}")
        print(f"  Total samples : {self.total}")
        print(f"  Passed        : {self.passed}  Failed: {self.failed}")
        print(f"")
        print(f"  Faithfulness      : {self.avg_faithfulness:.3f}  "
              f"(threshold={self.thresholds.get('faithfulness', '?')})")
        print(f"  Answer Relevance  : {self.avg_answer_relevance:.3f}  "
              f"(threshold={self.thresholds.get('answer_relevance', '?')})")
        print(f"  Context Recall    : {self.avg_context_recall:.3f}  "
              f"(threshold={self.thresholds.get('context_recall', '?')})")
        print(f"  Citation Coverage : {self.avg_citation_coverage:.3f}  "
              f"(threshold={self.thresholds.get('citation_coverage', '?')})")
        if self.avg_tool_selection is not None:
            print(f"  Tool Selection    : {self.avg_tool_selection:.3f}  "
                  f"(threshold={self.thresholds.get('tool_selection', '?')})")
        if self.threshold_breaches:
            print(f"\n  ⚠ Breached thresholds: {', '.join(self.threshold_breaches)}")
        print(f"{'─'*60}\n")


# ── Scorer ────────────────────────────────────────────────────────────────────

class FaithfulnessScorer:
    """
    Scores RAG answers using an LLM judge.
    All prompts are loaded from the versioned config file.
    """

    CITATION_PATTERN = re.compile(r"\[([a-f0-9]{8}:\d+)\]")

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            raise RuntimeError(f"OpenAI client failed to init: {e}")

    # ── Public API ────────────────────────────────────────────────────────────
    def score_sample(
        self,
        sample_id: str,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        expected_source: Optional[str] = None,
        actual_tools: Optional[List[str]] = None,
    ) -> SampleScore:
        """Score a single QA pair across all metrics including tool selection if expected_source is provided."""
        result = SampleScore(
            id=sample_id,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            expected_source=expected_source,
            actual_tools=actual_tools or [],
        )

        try:
            # 1. Faithfulness
            faith = self._score_faithfulness(answer, context)
            result.faithfulness         = faith.get("faithfulness_score", 0.0)
            result.faithfulness_claims  = faith.get("claims", [])
            result.faithfulness_reasoning = faith.get("reasoning", "")

            # 2. Answer relevance
            rel = self._score_relevance(question, answer)
            result.answer_relevance   = rel.get("relevance_score", 0.0)
            result.relevance_reasoning = rel.get("reasoning", "")

            # 3. Context recall
            recall = self._score_recall(ground_truth, context)
            result.context_recall    = recall.get("recall_score", 0.0)
            result.recall_reasoning  = recall.get("reasoning", "")

            # 4. Citation coverage (deterministic — no LLM needed)
            result.citation_coverage = self._score_citations(answer)

            # 5. Tool selection metric (if expected_source is provided)
            if expected_source:
                result.tool_selection_accuracy = self.score_tool_selection(
                    actual_tools=actual_tools or [],
                    expected_source=expected_source,
                )

        except Exception as e:
            result.error = str(e)

        return result

    def score_batch(
        self,
        samples: List[dict],
        max_workers: int = 4,
    ) -> List[SampleScore]:
        """
        Score a list of dicts, each containing:
          id, question, answer, context, ground_truth, (optional) expected_source, (optional) actual_tools
        """
        results: List[SampleScore] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self.score_sample,
                    s["id"], s["question"], s["answer"],
                    s["context"], s["ground_truth"],
                    s.get("expected_source"), s.get("actual_tools"),
                ): s["id"]
                for s in samples
            }
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r.id)
        return results

    # ── Private: LLM calls & Tool Selection ───────────────────────────────────

    @staticmethod
    def score_tool_selection(actual_tools: List[str], expected_source: Optional[str]) -> Optional[float]:
        if not expected_source:
            return None

        has_local = any(t == "search_local_documents" or "rag_query" in t for t in actual_tools)
        has_web = any(t != "search_local_documents" and "rag_query" not in t for t in actual_tools)

        exp = expected_source.lower()
        if exp == "local":
            return 1.0 if has_local and not has_web else 0.0
        elif exp == "web":
            return 1.0 if has_web and not has_local else 0.0
        elif exp == "both":
            return 1.0 if has_local and has_web else 0.0
        elif exp == "either":
            return 1.0 if (has_local or has_web) else 0.0
        return 0.0

    def _call_judge(self, prompt: str) -> dict:
        """Call the judge LLM and parse JSON response."""
        eval_cfg = self.cfg.evaluation
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=eval_cfg.scorer.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def _score_faithfulness(self, answer: str, context: str) -> dict:
        prompt = self.cfg.prompts.faithfulness_judge.format(
            context=context, answer=answer
        )
        return self._call_judge(prompt)

    def _score_relevance(self, question: str, answer: str) -> dict:
        prompt = self.cfg.prompts.answer_relevance_judge.format(
            question=question, answer=answer
        )
        return self._call_judge(prompt)

    def _score_recall(self, ground_truth: str, context: str) -> dict:
        prompt = self.cfg.prompts.context_recall_judge.format(
            ground_truth=ground_truth, context=context
        )
        return self._call_judge(prompt)

    def _score_citations(self, answer: str) -> float:
        """
        Deterministic: 1.0 if answer contains ≥1 valid citation pattern, else 0.0.
        Does not verify IDs are real — citation_enforcer handles that upstream.
        """
        return 1.0 if self.CITATION_PATTERN.search(answer) else 0.0


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    scores: List[SampleScore],
    thresholds: dict,
) -> EvalReport:
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot build report from empty score list")

    avg_faith   = sum(s.faithfulness     for s in scores) / n
    avg_rel     = sum(s.answer_relevance  for s in scores) / n
    avg_recall  = sum(s.context_recall    for s in scores) / n
    avg_cit     = sum(s.citation_coverage for s in scores) / n

    ts_scores = [s.tool_selection_accuracy for s in scores if s.tool_selection_accuracy is not None]
    avg_ts = sum(ts_scores) / len(ts_scores) if ts_scores else None

    breaches = []
    if avg_faith  < thresholds.get("faithfulness", 0.0):
        breaches.append("faithfulness")
    if avg_rel    < thresholds.get("answer_relevance", 0.0):
        breaches.append("answer_relevance")
    if avg_recall < thresholds.get("context_recall", 0.0):
        breaches.append("context_recall")
    if avg_cit    < thresholds.get("citation_coverage", 0.0):
        breaches.append("citation_coverage")
    if avg_ts is not None and avg_ts < thresholds.get("tool_selection", 0.0):
        breaches.append("tool_selection")

    passed = sum(1 for s in scores if s.passed(thresholds))

    return EvalReport(
        total=n,
        passed=passed,
        failed=n - passed,
        avg_faithfulness=avg_faith,
        avg_answer_relevance=avg_rel,
        avg_context_recall=avg_recall,
        avg_citation_coverage=avg_cit,
        avg_tool_selection=avg_ts,
        thresholds=thresholds,
        threshold_breaches=breaches,
        samples=scores,
    )
