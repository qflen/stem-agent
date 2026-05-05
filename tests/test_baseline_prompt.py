"""Baseline prompt must ask for JSON so its F1 reflects judgement, not parsing.

Until task 1 of the audit closeout, the undifferentiated prompt was free-form
prose; the live baseline F1 was 0.000 not because the LLM was incompetent but
because the parser dropped non-JSON responses on the floor. Folding the output
format into the baseline prompt closes that gap and makes the
specialized-vs-baseline delta a comparison of *judgement*, not formatting.
"""

from __future__ import annotations

import json

from stem_agent.capabilities.prompt_library import (
    OUTPUT_FORMAT_FRAGMENT,
    UNDIFFERENTIATED_PROMPT,
    compose_system_prompt,
)
from stem_agent.evaluation.benchmark import parse_review_response


def test_baseline_includes_output_format_header() -> None:
    assert "## Output Format" in UNDIFFERENTIATED_PROMPT


def test_baseline_specifies_is_clean_key() -> None:
    assert "is_clean" in UNDIFFERENTIATED_PROMPT


def test_baseline_specifies_issues_array() -> None:
    assert '"issues"' in UNDIFFERENTIATED_PROMPT


def test_baseline_does_not_include_capability_fragments() -> None:
    """The baseline stays untooled; only the output format discipline crosses over."""
    assert "## Structural Analysis Pass" not in UNDIFFERENTIATED_PROMPT
    assert "## Logic Correctness Pass" not in UNDIFFERENTIATED_PROMPT
    assert "## Security Analysis Pass" not in UNDIFFERENTIATED_PROMPT


def test_baseline_json_parses_through_review_parser() -> None:
    """A canonical clean-JSON response keyed off the baseline prompt round-trips."""
    response = json.dumps(
        {
            "issues": [],
            "summary": "No issues found.",
            "is_clean": True,
        }
    )
    parsed = parse_review_response(response)
    assert parsed.is_clean is True
    assert parsed.issues == []


def test_specialized_prompt_still_has_its_own_output_format() -> None:
    """``compose_system_prompt`` already appends OUTPUT_FORMAT_FRAGMENT;
    the baseline change must not duplicate it on the specialized side."""
    specialized = compose_system_prompt(["logic_correctness"])
    assert specialized.count(OUTPUT_FORMAT_FRAGMENT) == 1
