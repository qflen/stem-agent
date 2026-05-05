"""Deterministic rollback-loop demonstration; captures a journal for the writeup.

The live OpenAI run in ``docs/example_run/`` lands on ``SPECIALIZED`` on the
first attempt: useful to prove the happy path with real numbers, but it does
not exercise the closed-loop rollback branch the writeup spends its cross-check
section on. This script fills that gap **without spending tokens**: a scripted
LLM returns deliberately poor reviews on the first specialized attempt, trips
both the F1-threshold guard and the two cross-check disagreement signals, lets
``diagnose_failure`` derive its adjustments, and returns correct reviews once
those adjustments land in the composed prompt on the second attempt.

The resulting ``journal.json`` is a real journal written by real phase code:
the same ``ValidationPhase``, ``SpecializationPhase``, ``StateMachine``, and
``EvolutionJournal`` the live run uses. Only the LLM at the boundary is
scripted; clearly labelled as such in the companion README so nothing here
can be mistaken for an OpenAI run.

Usage:
    .venv/bin/python scripts/generate_rollback_demo.py

Output:
    docs/example_run_rollback/journal.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from stem_agent.adapters.json_storage import JsonStorageAdapter
from stem_agent.capabilities.prompt_library import UNDIFFERENTIATED_PROMPT
from stem_agent.core.agent import StemAgent
from stem_agent.core.config import StemAgentConfig

_ADJUSTMENT_MARKER = "IMPORTANT adjustments based on prior evaluation"
_BASELINE_MARKER = UNDIFFERENTIATED_PROMPT.splitlines()[0]


def _sensing_response() -> dict[str, Any]:
    return {
        "domain_name": "code_quality_analysis",
        "review_strategies": [
            "Multi-pass review with increasing specificity",
            "Start with structural analysis, then logic, then security",
            "Compare against known anti-patterns and best practices",
            "Use severity-based triage to prioritize findings",
        ],
        "issue_taxonomy": {
            "logic": ["off-by-one errors", "null pointer dereferences", "wrong operators"],
            "security": ["SQL injection", "path traversal", "hardcoded credentials"],
            "structure": ["high complexity", "deep nesting", "dead code"],
            "performance": ["N+1 queries", "unnecessary copies"],
        },
        "tool_categories": [
            "AST-based analysis",
            "Pattern matching / regex",
            "Complexity metrics",
        ],
        "output_format_patterns": [
            "Structured JSON with severity, category, line number, description",
        ],
        "key_insights": [
            "Expert reviewers distinguish false positives from real issues",
            "Multi-pass reviews catch more issues than single-pass",
            "Clean code that looks suspicious should not be flagged",
        ],
    }


def _planning_response() -> dict[str, Any]:
    return {
        "selected_capabilities": [
            "structural_analysis",
            "logic_correctness",
            "security_analysis",
            "performance_analysis",
            "severity_ranking",
        ],
        "review_passes": [
            {
                "pass_name": "structural",
                "focus_area": "Code structure, complexity, and maintainability",
                "capability_name": "structural_analysis",
                "priority": 1,
            },
            {
                "pass_name": "logic",
                "focus_area": "Logic correctness and edge cases",
                "capability_name": "logic_correctness",
                "priority": 2,
            },
            {
                "pass_name": "security",
                "focus_area": "Security vulnerabilities",
                "capability_name": "security_analysis",
                "priority": 3,
            },
            {
                "pass_name": "performance",
                "focus_area": "Performance issues and resource management",
                "capability_name": "performance_analysis",
                "priority": 4,
            },
        ],
        "evaluation_criteria": {"f1_threshold": 0.7, "precision_min": 0.5, "recall_min": 0.5},
        "domain_insights_for_prompt": (
            "Expert reviewers use multi-pass analysis, distinguishing real issues from "
            "false positives by understanding the code's context and intent."
        ),
        "reasoning": (
            "Selected structural, logic, security, performance, and severity passes "
            "to cover the full taxonomy."
        ),
    }


def _capability_generation_response() -> dict[str, Any]:
    """Benign proposal using only allowlisted imports; passes the sandbox."""
    return {
        "name": "input_validation_gap",
        "description": "Flag functions that accept outside data without guards.",
        "prompt_fragment": (
            "\n\n## Input Validation Gap Pass\n"
            "- Flag untyped input flowing into an index, arithmetic, or subscript."
        ),
        "validator_code": (
            "import re\n\n"
            "def check(code):\n"
            "    findings = []\n"
            "    if re.search(r'def [A-Za-z_][A-Za-z_0-9]*\\(', code) "
            "and 'isinstance' not in code and 'assert ' not in code:\n"
            "        findings.append('function defined without guards')\n"
            "    return findings\n"
        ),
    }


def _review(category: str | None, line: int = 1, description: str = "issue") -> str:
    if category is None:
        return json.dumps({"issues": [], "summary": "No issues found.", "is_clean": True})
    return json.dumps(
        {
            "issues": [
                {
                    "category": category,
                    "severity": "high",
                    "line_number": line,
                    "description": description,
                }
            ],
            "summary": f"{category} issue detected.",
            "is_clean": False,
        }
    )


# --- Attempt 1: deliberately weak reviews --------------------------------
#
# Goal: drop F1 below 0.7, trigger both cross-check failure modes.
#   * smell_03 is a short, shallow function; flagging it as structure
#     triggers ``llm_flagged_structure_but_ast_clean``.
#   * clean_02's EMAIL_PATTERN is safe; flagging it as structure triggers
#     the same cross-check again (second count feeds N in the adjustment).
#   * clean_03 uses ``eval`` safely with a whitelisted AST walker, but
#     the pattern scanner cannot distinguish "safe eval" from "unsafe
#     eval"; leaving clean_03 unflagged triggers
#     ``scanner_found_security_pattern_llm_missed``.
#
_BAD_REVIEWS_BY_CODE_KEY: dict[str, str] = {
    # Correctly flagged
    "def binary_search": _review("logic", 3, "Off-by-one on high bound"),
    "SELECT * FROM users": _review("security", 6, "SQL injection via f-string"),
    "API_KEY =": _review("security", 3, "Hardcoded credential"),
    "def calculate(expression": _review("security", 4, "Unsafe eval on user input"),
    "def process_order": _review("structure", 1, "God function"),
    "def compute_stats": _review("structure", 1, "Function too complex"),  # cross-check trips
    "def get_order_summaries": _review("performance", 6, "N+1 query pattern"),
    # Missed bugs (false negatives)
    "def filter_adults": _review(None),
    "def get_user_email": _review(None),
    "def average": _review(None),
    "def can_access_resource": _review(None),
    "def read_user_file": _review(None),
    "def classify_risk": _review(None),
    "def calculate_shipping": _review(None),
    "def find_common_elements": _review(None),
    # Over-flags on clean code (false positives)
    "def find_first_long_line": _review("logic", 7, "Assignment-in-condition"),
    "EMAIL_PATTERN = re.compile": _review("structure", 5, "Regex too complex"),  # cross-check trips
    # Correctly clean
    "def safe_eval_expr": _review(None),  # eval pattern scanner still fires; cross-check trips
    "def resilient_parse": _review(None),
    "def find_connected_components": _review(None),
}


# --- Attempt 2: adjusted reviews -----------------------------------------
#
# After the adjustment fragment is spliced in, every buggy sample is caught
# and every clean sample is left alone. F1 = 1.0 crosses both guards.
#
_GOOD_REVIEWS_BY_CODE_KEY: dict[str, str] = {
    "def binary_search": _review("logic", 3, "Off-by-one on high bound"),
    "def filter_adults": _review("logic", 3, "Should be >= 18, not > 18"),
    "def get_user_email": _review("logic", 4, "Missing None check"),
    "def average": _review("logic", 3, "Integer division loses precision"),
    "def can_access_resource": _review("logic", 9, "Boolean precedence bug"),
    "SELECT * FROM users": _review("security", 6, "SQL injection via f-string"),
    "def read_user_file": _review("security", 5, "Path traversal"),
    "API_KEY =": _review("security", 3, "Hardcoded credential"),
    "def calculate(expression": _review("security", 4, "Unsafe eval on user input"),
    "def process_order": _review("structure", 1, "God function"),
    "def classify_risk": _review("structure", 4, "Deep nesting with magic numbers"),
    "def compute_stats": _review("structure", 14, "Dead code after return"),
    "def calculate_shipping": _review("structure", 3, "Magic numbers"),
    "def get_order_summaries": _review("performance", 6, "N+1 query pattern"),
    "def find_common_elements": _review("performance", 7, "Unnecessary list copies"),
    "def find_first_long_line": _review(None),
    "EMAIL_PATTERN = re.compile": _review(None),
    "def safe_eval_expr": _review(None),
    "def resilient_parse": _review(None),
    "def find_connected_components": _review(None),
}


_STRUCTURED_RESPONSES: dict[str, dict[str, Any]] = {
    "understand the domain": _sensing_response(),
    "plan its specialization": _planning_response(),
    "propose ONE new review capability": _capability_generation_response(),
    "default": _sensing_response(),
}


class ScriptedLLM:
    """Deterministic LLM that models a weak-then-strong specialization.

    Baseline reviews (undifferentiated prompt) always return "clean" in
    free-form text so the baseline F1 is 0; mirroring the live-run
    parser artefact. Specialized reviews switch between two tables based
    on whether the adjustment marker is already in the prompt, which is
    exactly the signal the SpecializationPhase splices in post-rollback.
    """

    last_usage: dict[str, int] | None

    def __init__(self) -> None:
        self.last_usage = None
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _fake_usage(prompt: str, response: str) -> dict[str, int]:
        p = max(1, len(prompt) // 4)
        c = max(1, len(response) // 4)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        self.calls.append({"method": "generate", "model": model})
        if _BASELINE_MARKER in prompt and "## Domain-Specific Insights" not in prompt:
            response = "The code looks fine to me. No issues found."
        else:
            table = (
                _GOOD_REVIEWS_BY_CODE_KEY
                if _ADJUSTMENT_MARKER in prompt
                else _BAD_REVIEWS_BY_CODE_KEY
            )
            response = _review(None)
            for key, candidate in table.items():
                if key in prompt:
                    response = candidate
                    break
        self.last_usage = self._fake_usage(prompt, response)
        return response

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        self.calls.append({"method": "structured_generate", "model": model})
        for key, data in _STRUCTURED_RESPONSES.items():
            if key in prompt:
                self.last_usage = self._fake_usage(prompt, json.dumps(data))
                return response_model.model_validate(data)
        fallback = _STRUCTURED_RESPONSES["default"]
        self.last_usage = self._fake_usage(prompt, json.dumps(fallback))
        return response_model.model_validate(fallback)


def main() -> None:
    config = StemAgentConfig(
        openai_api_key="demo-key",
        planning_model="scripted-planner",
        execution_model="scripted-executor",
        f1_threshold=0.7,
        improvement_required=True,
        max_rollback_attempts=3,
    )

    llm = ScriptedLLM()
    storage = JsonStorageAdapter(config.journal_dir)
    agent = StemAgent(config=config, llm=llm, storage=storage)

    succeeded = agent.differentiate(domain="code_quality_analysis")

    out_dir = Path("docs/example_run_rollback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "journal.json"
    out_path.write_text(json.dumps(agent.journal.to_dict(), indent=2, default=str))

    # A one-line receipt; the README next to the journal has the full breakdown.
    print(f"\nSpecialization outcome: {'SPECIALIZED' if succeeded else 'FAILED'}")
    print(f"Final state: {agent.state.value}")
    print(f"Journal events: {len(agent.journal)}")
    print(f"Journal written to: {out_path}")


if __name__ == "__main__":
    main()
