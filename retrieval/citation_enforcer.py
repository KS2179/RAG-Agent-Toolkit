"""
retrieval/citation_enforcer.py
Post-generation citation validation and retry loop.

After the LLM produces an answer, this module:
  1. Scans the answer for inline citations matching [xxxxxxxx:N]
  2. Verifies each citation ID exists in the retrieved chunks
  3. If violations are found → retry / warn / raise (configurable)

This is the "citation enforcement" component from Phase 2 spec.
All rules live in config/phase2.yaml under citation_enforcement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from config.loader import CitationEnforcementConfig
from store.vector_store import RetrievedChunk


@dataclass
class CitationCheckResult:
    answer: str
    is_valid: bool
    found_citations: List[str]          
    invalid_citations: List[str]        
    missing_count: int                  
    violation_messages: List[str] = field(default_factory=list)

WEB_PATTERN = re.compile(r"\[web:(\d+)\]")

class CitationEnforcer:
    """
    Validates that an LLM answer contains well-formed inline citations
    that reference the actually-retrieved chunks.
    """

    def __init__(self, cfg: CitationEnforcementConfig):
        self.cfg = cfg
        self._pattern = re.compile(cfg.citation_pattern)

    def check(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> CitationCheckResult:
        """
        Check an answer against the retrieved chunks.

        Returns a CitationCheckResult with is_valid=True only when:
          - At least min_citations_required inline citations are present
          - All cited IDs belong to the retrieved set (no hallucinated refs)
        """
        valid_ids: Set[str] = {c.citation_id for c in retrieved_chunks}
        found = self._pattern.findall(answer)
        found_unique = list(dict.fromkeys(found))   

        invalid = [cid for cid in found_unique if cid not in valid_ids]
        missing = max(0, self.cfg.min_citations_required - len(found_unique))

        violations: List[str] = []
        if missing > 0:
            violations.append(
                f"Answer contains {len(found_unique)} citation(s); "
                f"at least {self.cfg.min_citations_required} required."
            )
        if invalid:
            violations.append(
                f"Hallucinated citation IDs (not in retrieved set): {invalid}"
            )

        return CitationCheckResult(
            answer=answer,
            is_valid=(len(violations) == 0),
            found_citations=found_unique,
            invalid_citations=invalid,
            missing_count=missing,
            violation_messages=violations,
        )

    def check_agentic(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        web_sources: Optional[List[dict]] = None,
    ) -> CitationCheckResult:
        """
        Check an agentic answer supporting both local [doc_id:chunk_index] and web [web:n] citations.
        """
        valid_local_ids: Set[str] = {c.citation_id for c in retrieved_chunks}
        local_found = self._pattern.findall(answer)
        web_found_indices = WEB_PATTERN.findall(answer)

        local_found_unique = list(dict.fromkeys(local_found))
        web_found_unique = list(dict.fromkeys(web_found_indices))

        invalid_local = [cid for cid in local_found_unique if cid not in valid_local_ids]

        num_web_sources = len(web_sources or [])
        invalid_web = [
            f"web:{idx}" for idx in web_found_unique
            if not (idx.isdigit() and 1 <= int(idx) <= num_web_sources)
        ]

        total_citations = len(local_found_unique) + len(web_found_unique)
        missing = max(0, self.cfg.min_citations_required - total_citations)

        violations: List[str] = []
        if missing > 0:
            violations.append(
                f"Answer contains {total_citations} citation(s); "
                f"at least {self.cfg.min_citations_required} required."
            )
        if invalid_local:
            violations.append(f"Hallucinated local citation IDs: {invalid_local}")
        if invalid_web:
            violations.append(f"Hallucinated web citation IDs: {invalid_web}")

        all_found = [f"[{cid}]" for cid in local_found_unique] + [f"[web:{idx}]" for idx in web_found_unique]
        all_invalid = invalid_local + invalid_web

        return CitationCheckResult(
            answer=answer,
            is_valid=(len(violations) == 0),
            found_citations=all_found,
            invalid_citations=all_invalid,
            missing_count=missing,
            violation_messages=violations,
        )

    def enforce(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        retry_fn=None,              
    ) -> CitationCheckResult:
        """
        Check and — if configured — retry or raise on violation.

        Args:
            answer:           Initial LLM answer.
            retrieved_chunks: Chunks used for this answer.
            retry_fn:         Optional callable that takes the previous answer
                              and returns a new answer string. Used when
                              on_violation == "retry".

        Returns:
            The final CitationCheckResult (possibly after retries).
        """
        if not self.cfg.enabled:
            return CitationCheckResult(
                answer=answer,
                is_valid=True,
                found_citations=self._pattern.findall(answer),
                invalid_citations=[],
                missing_count=0,
            )

        result = self.check(answer, retrieved_chunks)

        if result.is_valid:
            return result

        if self.cfg.on_violation == "raise":
            raise CitationViolationError(result)

        if self.cfg.on_violation == "warn":
            import warnings
            warnings.warn(
                "Citation violation: " + "; ".join(result.violation_messages),
                stacklevel=2,
            )
            return result

        if self.cfg.on_violation == "retry" and retry_fn is not None:
            for attempt in range(self.cfg.max_retries):
                new_answer = retry_fn(result.answer)
                result = self.check(new_answer, retrieved_chunks)
                if result.is_valid:
                    return result
            import warnings
            warnings.warn(
                f"Citation enforcement: {self.cfg.max_retries} retries exhausted. "
                "Returning best available answer.",
                stacklevel=2,
            )

        return result


class CitationViolationError(Exception):
    def __init__(self, result: CitationCheckResult):
        self.result = result
        super().__init__(
            "Citation enforcement failed:\n"
            + "\n".join(f"  • {v}" for v in result.violation_messages)
        )
