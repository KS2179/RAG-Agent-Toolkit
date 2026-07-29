"""
tests/test_phase3.py
Unit tests for Phase 3:
  - Dataset validation (schema, duplicates, edge cases)
  - FaithfulnessScorer (mocked LLM — no API calls)
  - Report builder (thresholds, breach detection, CI gate)
  - Citation coverage scoring (deterministic)
  - EvalReport.ci_passed logic
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_cfg():
    """Minimal PipelineConfig with Phase 3 eval settings."""
    from config.loader import (
        PipelineConfig, ChunkingConfig, EmbeddingConfig,
        VectorStoreConfig, RetrievalConfig, PromptsConfig,
        CitationEnforcementConfig,
    )
    from pydantic import BaseModel
    from typing import Optional

    # Inline eval config to avoid file I/O
    class ScorerCfg(BaseModel):
        model: str = "gpt-4o-mini"
        batch_size: int = 10
        max_workers: int = 2

    class CICfg(BaseModel):
        fail_on_threshold_breach: bool = True
        min_dataset_size: int = 5
        sample_size: Optional[int] = None

    class ThresholdsCfg(BaseModel):
        faithfulness: float = 0.75
        answer_relevance: float = 0.70
        context_recall: float = 0.65
        citation_coverage: float = 0.80

    class EvalCfg(BaseModel):
        dataset_path: str = "eval/dataset.jsonl"
        results_path: str = "eval/results.jsonl"
        report_path: str = "eval/report.json"
        thresholds: ThresholdsCfg = ThresholdsCfg()
        scorer: ScorerCfg = ScorerCfg()
        ci: CICfg = CICfg()

    cfg = MagicMock()
    cfg.evaluation = EvalCfg()
    cfg.prompts.faithfulness_judge   = "Context: {context}\nAnswer: {answer}"
    cfg.prompts.answer_relevance_judge = "Question: {question}\nAnswer: {answer}"
    cfg.prompts.context_recall_judge = "GT: {ground_truth}\nContext: {context}"
    return cfg


@pytest.fixture
def sample_scores():
    from eval.scorer import SampleScore
    return [
        SampleScore("q001", "Q1", "A1 [abcd1234:0]", "GT1",
                    faithfulness=0.9, answer_relevance=0.85,
                    context_recall=0.80, citation_coverage=1.0),
        SampleScore("q002", "Q2", "A2", "GT2",
                    faithfulness=0.6, answer_relevance=0.5,
                    context_recall=0.4, citation_coverage=0.0),
        SampleScore("q003", "Q3", "A3 [efgh5678:1]", "GT3",
                    faithfulness=0.8, answer_relevance=0.75,
                    context_recall=0.70, citation_coverage=1.0),
    ]


# ── Dataset validation ────────────────────────────────────────────────────────

class TestDatasetValidation:

    def _valid_record(self, id_="q001"):
        return {
            "id": id_,
            "question": "What is artificial intelligence?",
            "ground_truth": "AI is intelligence demonstrated by machines.",
            "source_docs": ["ai_overview.txt"],
            "difficulty": "easy",
            "category": "factual",
        }

    def test_valid_record_passes(self):
        from eval.build_dataset import validate_record
        errors = validate_record(self._valid_record(), 0)
        assert errors == []

    def test_missing_field_raises_error(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        del record["ground_truth"]
        errors = validate_record(record, 0)
        assert any("ground_truth" in e for e in errors)

    def test_invalid_difficulty(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        record["difficulty"] = "impossible"
        errors = validate_record(record, 0)
        assert any("difficulty" in e for e in errors)

    def test_invalid_category(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        record["category"] = "nonsense"
        errors = validate_record(record, 0)
        assert any("category" in e for e in errors)

    def test_question_too_short(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        record["question"] = "AI?"
        errors = validate_record(record, 0)
        assert any("short" in e for e in errors)

    def test_all_60_seed_records_valid(self):
        from eval.build_dataset import validate_record, SEED_DATASET
        all_errors = []
        for i, record in enumerate(SEED_DATASET):
            all_errors.extend(validate_record(record, i))
        assert all_errors == [], f"Seed dataset errors: {all_errors}"

    def test_seed_dataset_has_no_duplicate_ids(self):
        from eval.build_dataset import SEED_DATASET
        ids = [r["id"] for r in SEED_DATASET]
        assert len(ids) == len(set(ids))

    def test_seed_dataset_size(self):
        from eval.build_dataset import SEED_DATASET
        assert len(SEED_DATASET) >= 50

    def test_seed_has_all_categories(self):
        from eval.build_dataset import SEED_DATASET
        cats = {r["category"] for r in SEED_DATASET}
        assert cats == {"factual", "reasoning", "multi-hop", "edge-case"}

    def test_seed_has_all_difficulties(self):
        from eval.build_dataset import SEED_DATASET
        diffs = {r["difficulty"] for r in SEED_DATASET}
        assert diffs == {"easy", "medium", "hard"}


# ── Citation coverage (deterministic) ────────────────────────────────────────

class TestCitationCoverage:

    def _scorer(self, cfg):
        from eval.scorer import FaithfulnessScorer
        scorer = FaithfulnessScorer.__new__(FaithfulnessScorer)
        scorer.cfg = cfg
        return scorer

    def test_answer_with_citation_scores_1(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("Answer here [abcd1234:0].") == 1.0

    def test_answer_without_citation_scores_0(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("No citations here.") == 0.0

    def test_multiple_citations_still_scores_1(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("[abcd1234:0] and [efgh5678:1]") == 1.0

    def test_malformed_citation_scores_0(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("See [notvalid] for details.") == 0.0


# ── Scorer (mocked LLM) ───────────────────────────────────────────────────────

class TestFaithfulnessScorer:

    def _build_scorer(self, minimal_cfg):
        from eval.scorer import FaithfulnessScorer
        scorer = FaithfulnessScorer.__new__(FaithfulnessScorer)
        scorer.cfg = minimal_cfg

        mock_client = MagicMock()
        scorer._client = mock_client

        def fake_judge(prompt: str) -> dict:
            # Check GT: first — the recall prompt contains both "GT:" and
            # "Context:", so it must be matched before the faithfulness branch.
            if "GT:" in prompt:
                return {"recall_score": 0.75, "reasoning": "Most ground truth is covered."}
            elif "Question:" in prompt:
                return {"relevance_score": 0.85, "reasoning": "Directly answers the question."}
            else:
                # Faithfulness prompt: "Context: {context}\nAnswer: {answer}"
                return {
                    "faithfulness_score": 0.9,
                    "claims": [{"claim": "AI is intelligent", "supported": True, "evidence": "..."}],
                    "reasoning": "All claims supported.",
                }

        scorer._call_judge = fake_judge
        return scorer

    def test_score_sample_returns_all_metrics(self, minimal_cfg):
        scorer = self._build_scorer(minimal_cfg)
        result = scorer.score_sample(
            "q001", "What is AI?", "AI is intelligent machines [abcd1234:0].",
            "AI context here.", "AI is intelligence by machines."
        )
        assert result.faithfulness     > 0
        assert result.answer_relevance > 0
        assert result.context_recall   > 0
        assert result.citation_coverage == 1.0
        assert result.error is None

    def test_score_sample_no_citation(self, minimal_cfg):
        scorer = self._build_scorer(minimal_cfg)
        result = scorer.score_sample(
            "q002", "What is AI?", "AI is intelligent machines.",
            "context", "ground truth"
        )
        assert result.citation_coverage == 0.0

    def test_score_sample_handles_error_gracefully(self, minimal_cfg):
        scorer = self._build_scorer(minimal_cfg)
        scorer._call_judge = MagicMock(side_effect=Exception("API error"))
        result = scorer.score_sample("q003", "Q?", "A.", "ctx", "gt")
        assert result.error is not None
        assert result.faithfulness == 0.0


# ── Report builder ────────────────────────────────────────────────────────────

class TestReportBuilder:

    def test_report_averages_correct(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.75, "answer_relevance": 0.70,
                      "context_recall": 0.65, "citation_coverage": 0.80}
        report = build_report(sample_scores, thresholds)
        assert abs(report.avg_faithfulness - (0.9 + 0.6 + 0.8) / 3) < 1e-6
        assert report.total == 3

    def test_threshold_breach_detected(self, sample_scores):
        from eval.scorer import build_report
        # Set high thresholds to force breaches
        thresholds = {"faithfulness": 0.99, "answer_relevance": 0.99,
                      "context_recall": 0.99, "citation_coverage": 0.99}
        report = build_report(sample_scores, thresholds)
        assert len(report.threshold_breaches) > 0
        assert not report.ci_passed

    def test_all_pass_when_thresholds_low(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.0, "answer_relevance": 0.0,
                      "context_recall": 0.0, "citation_coverage": 0.0}
        report = build_report(sample_scores, thresholds)
        assert report.ci_passed

    def test_empty_scores_raises(self):
        from eval.scorer import build_report
        with pytest.raises(ValueError):
            build_report([], {})

    def test_passed_count(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.75, "answer_relevance": 0.70,
                      "context_recall": 0.65, "citation_coverage": 0.80}
        report = build_report(sample_scores, thresholds)
        # q001 and q003 should pass; q002 should fail
        assert report.passed == 2
        assert report.failed == 1

    def test_report_to_dict_includes_ci_passed(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.0, "answer_relevance": 0.0,
                      "context_recall": 0.0, "citation_coverage": 0.0}
        report = build_report(sample_scores, thresholds)
        d = report.to_dict()
        assert "ci_passed" in d
        assert d["ci_passed"] is True
