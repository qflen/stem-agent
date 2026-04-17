"""Benchmark runner — feeds code samples through an agent and collects verdicts.

The runner is agent-agnostic: it takes a review function and a corpus,
runs each sample, and computes metrics. This lets us benchmark both
the undifferentiated baseline and the specialized agent identically.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from stem_agent.core.journal import EvolutionJournal
from stem_agent.evaluation.fixtures.code_samples import BenchmarkSample, get_benchmark_corpus
from stem_agent.evaluation.metrics import ClassificationMetrics, compute_metrics
from stem_agent.ports.llm import LLMPort


class ReviewResult(BaseModel):
    """Structured output from a code review."""

    class Issue(BaseModel):
        category: str
        severity: str
        line_number: int = 0
        description: str = ""
        suggestion: str = ""

    issues: list[Issue] = Field(default_factory=list)
    summary: str = ""
    is_clean: bool = True


@dataclass
class SampleVerdict:
    """The result of reviewing a single benchmark sample."""

    sample_id: str
    detected_categories: set[str]
    is_clean_detected: bool
    ground_truth_categories: set[str]
    is_clean_truth: bool
    raw_issues: list[dict[str, Any]] = field(default_factory=list)


# Category normalization: map various LLM outputs to canonical categories
CATEGORY_ALIASES: dict[str, str] = {
    "bug": "logic",
    "bugs": "logic",
    "logic_bug": "logic",
    "logic_bugs": "logic",
    "logic_error": "logic",
    "correctness": "logic",
    "off_by_one": "logic",
    "off-by-one": "logic",
    "null_check": "logic",
    "security_vulnerability": "security",
    "security_vulnerabilities": "security",
    "vulnerability": "security",
    "injection": "security",
    "sql_injection": "security",
    "path_traversal": "security",
    "hardcoded_credentials": "security",
    "credentials": "security",
    "code_smell": "structure",
    "code_smells": "structure",
    "maintainability": "structure",
    "complexity": "structure",
    "dead_code": "structure",
    "style": "structure",
    "performance_issue": "performance",
    "performance_issues": "performance",
    "efficiency": "performance",
    "n_plus_one": "performance",
    "n+1": "performance",
}

CANONICAL_CATEGORIES = {"logic", "security", "structure", "performance"}


def normalize_category(raw: str) -> str:
    """Normalize a category string to one of the canonical categories."""
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in CANONICAL_CATEGORIES:
        return cleaned
    return CATEGORY_ALIASES.get(cleaned, cleaned)


def parse_review_response(raw_response: str) -> ReviewResult:
    """Parse an LLM review response into a structured ReviewResult.

    Handles both clean JSON and JSON embedded in markdown code blocks.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last lines (fences)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # If we can't parse JSON, treat as a text response indicating clean code
        # only if no issue-like keywords are present
        lower = text.lower()
        has_issues = any(
            kw in lower for kw in ["bug", "issue", "vulnerability", "error", "problem"]
        )
        return ReviewResult(
            issues=[],
            summary=text[:200],
            is_clean=not has_issues,
        )

    issues = []
    for item in data.get("issues", []):
        issues.append(
            ReviewResult.Issue(
                category=normalize_category(item.get("category", "unknown")),
                severity=item.get("severity", "medium").lower(),
                line_number=item.get("line_number", 0),
                description=item.get("description", ""),
                suggestion=item.get("suggestion", ""),
            )
        )

    return ReviewResult(
        issues=issues,
        summary=data.get("summary", ""),
        is_clean=data.get("is_clean", len(issues) == 0),
    )


# Type for a review function: takes code + system prompt, returns raw LLM response
ReviewFunction = Callable[[str, str], str]


def run_benchmark(
    review_fn: ReviewFunction,
    system_prompt: str,
    corpus: list[BenchmarkSample] | None = None,
) -> tuple[list[SampleVerdict], ClassificationMetrics]:
    """Run the full benchmark suite and compute metrics.

    Args:
        review_fn: Function that takes (code, system_prompt) and returns raw LLM text.
        system_prompt: The system prompt to use for reviews.
        corpus: Benchmark samples. Uses default corpus if None.

    Returns:
        Tuple of (per-sample verdicts, aggregate metrics).
    """
    if corpus is None:
        corpus = get_benchmark_corpus()

    verdicts: list[SampleVerdict] = []
    detected_cats_list: list[set[str]] = []
    truth_cats_list: list[set[str]] = []
    det_clean_list: list[bool] = []
    truth_clean_list: list[bool] = []

    for sample in corpus:
        raw_response = review_fn(sample.code, system_prompt)
        result = parse_review_response(raw_response)

        detected_cats = {issue.category for issue in result.issues}
        truth_cats = {normalize_category(cat) for cat in sample.issue_categories}

        verdict = SampleVerdict(
            sample_id=sample.sample_id,
            detected_categories=detected_cats,
            is_clean_detected=result.is_clean,
            ground_truth_categories=truth_cats,
            is_clean_truth=sample.is_clean,
            raw_issues=[i.model_dump() for i in result.issues],
        )
        verdicts.append(verdict)

        detected_cats_list.append(detected_cats)
        truth_cats_list.append(truth_cats)
        det_clean_list.append(result.is_clean)
        truth_clean_list.append(sample.is_clean)

    metrics = compute_metrics(
        detected_cats_list,
        truth_cats_list,
        det_clean_list,
        truth_clean_list,
    )
    return verdicts, metrics


def make_llm_review_fn(
    llm: LLMPort,
    model: str | None = None,
    *,
    journal: EvolutionJournal | None = None,
    phase: str = "validation",
) -> ReviewFunction:
    """Create a review function backed by an LLM adapter.

    If ``journal`` is supplied, each call is logged as an ``LLM_CALL`` event
    with prompt hash, model, and token count pulled from ``llm.last_usage``.

    Args:
        llm: An LLM adapter satisfying the LLMPort protocol.
        model: Optional model override.
        journal: If provided, log every review call to this journal.
        phase: Phase name attached to the logged event.

    Returns:
        A ReviewFunction compatible with run_benchmark.
    """

    def review_fn(code: str, system_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n## Code to Review\n```python\n{code}\n```"
        response = llm.generate(full_prompt, model=model)
        if journal is not None:
            usage = getattr(llm, "last_usage", None)
            journal.log_llm_call(
                phase=phase,
                model=model or "default",
                prompt_hash=EvolutionJournal.hash_prompt(full_prompt),
                token_count=usage.get("total_tokens") if usage else None,
            )
        return response

    return review_fn
