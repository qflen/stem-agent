"""Shared test fixtures — including FakeLLM, a first-class test double.

FakeLLM is not an afterthought: it implements the same LLM Protocol
as production adapters and returns realistic, structured responses.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest
from pydantic import BaseModel

from stem_agent.capabilities.registry import CapabilityRegistry, build_default_registry
from stem_agent.core.config import StemAgentConfig
from stem_agent.core.journal import EvolutionJournal
from stem_agent.core.state_machine import StateMachine
from stem_agent.evaluation.fixtures.code_samples import (
    BenchmarkSample,
    get_benchmark_corpus,
)


class FakeLLM:
    """Deterministic LLM test double implementing the LLMPort protocol.

    Returns pre-configured responses keyed by substring matches in the prompt.
    Falls back to a default response if no match is found. Tracks all calls
    for assertion.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        structured_responses: dict[str, dict[str, Any]] | None = None,
        default_response: str = '{"issues": [], "summary": "No issues found.", "is_clean": true}',
    ) -> None:
        self._responses = responses or {}
        self._structured_responses = structured_responses or {}
        self._default_response = default_response
        self.calls: list[dict[str, Any]] = []
        self.last_usage: dict[str, int] | None = None

    def _fake_usage(self, prompt: str, completion: str) -> dict[str, int]:
        """Return ~4-chars-per-token estimates so token_count is populated."""
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        """Return a canned response based on prompt substring matching."""
        self.calls.append({"method": "generate", "prompt": prompt, "model": model})

        for key, response in self._responses.items():
            if key in prompt:
                self.last_usage = self._fake_usage(prompt, response)
                return response
        self.last_usage = self._fake_usage(prompt, self._default_response)
        return self._default_response

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        """Return a structured response matching the requested model."""
        self.calls.append(
            {
                "method": "structured_generate",
                "prompt": prompt,
                "model": model,
                "response_model": response_model.__name__,
            }
        )

        for key, data in self._structured_responses.items():
            if key in prompt:
                self.last_usage = self._fake_usage(prompt, json.dumps(data))
                return response_model.model_validate(data)

        fallback = self._structured_responses.get("default", {})
        self.last_usage = self._fake_usage(prompt, json.dumps(fallback))
        return response_model.model_validate(fallback)


def _make_sensing_response() -> dict[str, Any]:
    """Realistic sensing response for code quality analysis."""
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
            "Complexity metrics (cyclomatic, cognitive)",
            "Security vulnerability scanners",
        ],
        "output_format_patterns": [
            "Structured JSON with severity, category, line number, description, suggestion",
            "Issues ordered by severity (critical first)",
            "Summary with overall assessment",
        ],
        "key_insights": [
            "Expert reviewers distinguish false positives from real issues",
            "Multi-pass reviews catch more issues than single-pass",
            "Clean code that looks suspicious should not be flagged — specificity matters",
        ],
    }


def _make_planning_response() -> dict[str, Any]:
    """Realistic planning response."""
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
        "evaluation_criteria": {
            "f1_threshold": 0.6,
            "precision_min": 0.5,
            "recall_min": 0.5,
        },
        "domain_insights_for_prompt": (
            "Expert code reviewers use multi-pass analysis, starting with "
            "structural properties before diving into logic and security. "
            "They distinguish real issues from false positives by understanding "
            "the code's context and intent."
        ),
        "reasoning": (
            "Selected all analysis capabilities plus severity ranking for comprehensive "
            "multi-pass review. Four review passes ordered by priority ensure no category "
            "is missed while keeping each pass focused."
        ),
    }


def _make_security_sensing_response() -> dict[str, Any]:
    """Sensing response scoped to security-audit work."""
    return {
        "domain_name": "security_audit",
        "review_strategies": [
            "Threat-model the code path before reading line by line",
            "Trace every sink for tainted input back to its source",
            "Treat any string-building SQL or shell command as hostile",
        ],
        "issue_taxonomy": {
            "security": [
                "SQL injection",
                "path traversal",
                "command injection",
                "hardcoded credentials",
                "insecure deserialization",
                "weak cryptography",
            ],
        },
        "tool_categories": [
            "Regex pattern matching for known unsafe constructs",
            "Taint analysis for sink/source tracking",
        ],
        "output_format_patterns": [
            "Structured JSON with severity, category, line number, exploit sketch, remediation",
        ],
        "key_insights": [
            "Most false positives come from whitelisted sinks that look unsafe at a glance",
            "A clean code sample may use eval/subprocess safely if the inputs are constrained",
        ],
    }


def _make_security_planning_response() -> dict[str, Any]:
    """Narrower planning response: only security + severity ranking."""
    return {
        "selected_capabilities": [
            "security_analysis",
            "severity_ranking",
        ],
        "review_passes": [
            {
                "pass_name": "security",
                "focus_area": "Security vulnerabilities and unsafe sinks",
                "capability_name": "security_analysis",
                "priority": 1,
            },
            {
                "pass_name": "severity",
                "focus_area": "Rank findings by exploitability",
                "capability_name": "severity_ranking",
                "priority": 2,
            },
        ],
        "evaluation_criteria": {
            "f1_threshold": 0.5,
            "precision_min": 0.5,
            "recall_min": 0.5,
        },
        "domain_insights_for_prompt": (
            "Focus exclusively on security: trace untrusted input to every sink. "
            "Do not flag style or structural issues — they are out of scope for an audit."
        ),
        "reasoning": (
            "A security audit benefits from a narrow, deep focus rather than a broad "
            "multi-pass review; omitting structural and style passes reduces false positives."
        ),
    }


def _make_review_response_with_issues() -> str:
    """Realistic review response detecting issues."""
    return json.dumps(
        {
            "issues": [
                {
                    "category": "logic",
                    "severity": "high",
                    "line_number": 3,
                    "description": "Off-by-one error in boundary check",
                    "suggestion": "Use len(arr) - 1 instead of len(arr)",
                },
            ],
            "summary": "Found 1 logic issue in the code.",
            "is_clean": False,
        }
    )


def _make_review_response_clean() -> str:
    """Review response for clean code."""
    return json.dumps(
        {
            "issues": [],
            "summary": "Code is well-structured and correct. No issues found.",
            "is_clean": True,
        }
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A FakeLLM with realistic pre-configured responses for all phases."""
    return FakeLLM(
        responses={
            # Reviews with issues
            "binary_search": _make_review_response_with_issues(),
            "filter_adults": json.dumps(
                {
                    "issues": [
                        {
                            "category": "logic",
                            "severity": "high",
                            "line_number": 3,
                            "description": "Should be >= 18",
                        }
                    ],
                    "summary": "Wrong operator.",
                    "is_clean": False,
                }
            ),
            "get_user_email": json.dumps(
                {
                    "issues": [
                        {
                            "category": "logic",
                            "severity": "high",
                            "line_number": 4,
                            "description": "Missing None check",
                        }
                    ],
                    "summary": "None check missing.",
                    "is_clean": False,
                }
            ),
            "average": json.dumps(
                {
                    "issues": [
                        {
                            "category": "logic",
                            "severity": "medium",
                            "line_number": 3,
                            "description": "Potential overflow",
                        }
                    ],
                    "summary": "Overflow risk.",
                    "is_clean": False,
                }
            ),
            "can_access_resource": json.dumps(
                {
                    "issues": [
                        {
                            "category": "logic",
                            "severity": "high",
                            "line_number": 9,
                            "description": "Wrong boolean precedence",
                        }
                    ],
                    "summary": "Boolean precedence bug.",
                    "is_clean": False,
                }
            ),
            "SELECT * FROM users": json.dumps(
                {
                    "issues": [
                        {
                            "category": "security",
                            "severity": "critical",
                            "line_number": 6,
                            "description": "SQL injection via f-string",
                        }
                    ],
                    "summary": "SQL injection.",
                    "is_clean": False,
                }
            ),
            "read_user_file": json.dumps(
                {
                    "issues": [
                        {
                            "category": "security",
                            "severity": "critical",
                            "line_number": 5,
                            "description": "Path traversal vulnerability",
                        }
                    ],
                    "summary": "Path traversal.",
                    "is_clean": False,
                }
            ),
            "API_KEY": json.dumps(
                {
                    "issues": [
                        {
                            "category": "security",
                            "severity": "critical",
                            "line_number": 3,
                            "description": "Hardcoded API key",
                        }
                    ],
                    "summary": "Hardcoded credentials.",
                    "is_clean": False,
                }
            ),
            "eval(expression)": json.dumps(
                {
                    "issues": [
                        {
                            "category": "security",
                            "severity": "critical",
                            "line_number": 4,
                            "description": "Unsafe eval with user input",
                        }
                    ],
                    "summary": "Unsafe eval.",
                    "is_clean": False,
                }
            ),
            "process_order": json.dumps(
                {
                    "issues": [
                        {
                            "category": "structure",
                            "severity": "medium",
                            "line_number": 1,
                            "description": "Function too long",
                        }
                    ],
                    "summary": "God function.",
                    "is_clean": False,
                }
            ),
            "classify_risk": json.dumps(
                {
                    "issues": [
                        {
                            "category": "structure",
                            "severity": "medium",
                            "line_number": 4,
                            "description": "Deep nesting with magic numbers",
                        }
                    ],
                    "summary": "Deep nesting.",
                    "is_clean": False,
                }
            ),
            "Dead code below": json.dumps(
                {
                    "issues": [
                        {
                            "category": "structure",
                            "severity": "medium",
                            "line_number": 13,
                            "description": "Unreachable code after return",
                        }
                    ],
                    "summary": "Dead code.",
                    "is_clean": False,
                }
            ),
            "calculate_shipping": json.dumps(
                {
                    "issues": [
                        {
                            "category": "structure",
                            "severity": "low",
                            "line_number": 3,
                            "description": "Magic numbers without constants",
                        }
                    ],
                    "summary": "Magic numbers.",
                    "is_clean": False,
                }
            ),
            "get_order_summaries": json.dumps(
                {
                    "issues": [
                        {
                            "category": "performance",
                            "severity": "high",
                            "line_number": 6,
                            "description": "N+1 query pattern",
                        }
                    ],
                    "summary": "N+1 queries.",
                    "is_clean": False,
                }
            ),
            "find_common_elements": json.dumps(
                {
                    "issues": [
                        {
                            "category": "performance",
                            "severity": "medium",
                            "line_number": 7,
                            "description": "Unnecessary list copies",
                        }
                    ],
                    "summary": "Unnecessary copies.",
                    "is_clean": False,
                }
            ),
            # Clean code — should report no issues
            "walrus": _make_review_response_clean(),
            "EMAIL_PATTERN": _make_review_response_clean(),
            "safe_eval_expr": _make_review_response_clean(),
            "resilient_parse": _make_review_response_clean(),
            "find_connected_components": _make_review_response_clean(),
        },
        structured_responses={
            # More-specific security_audit keys come first so substring matching
            # short-circuits before hitting the generic code-quality responses.
            "domain of security_audit": _make_security_sensing_response(),
            "specialization for security_audit": _make_security_planning_response(),
            "understand the domain": _make_sensing_response(),
            "plan its specialization": _make_planning_response(),
            "default": _make_sensing_response(),
        },
    )


@pytest.fixture
def poor_fake_llm() -> FakeLLM:
    """A FakeLLM that produces poor results — triggers rollback."""
    return FakeLLM(
        default_response=json.dumps(
            {
                "issues": [
                    {
                        "category": "style",
                        "severity": "low",
                        "line_number": 1,
                        "description": "Vague style issue",
                    },
                ],
                "summary": "Found a style issue.",
                "is_clean": False,
            }
        ),
        structured_responses={
            "understand the domain": _make_sensing_response(),
            "plan its specialization": _make_planning_response(),
            "default": _make_sensing_response(),
        },
    )


@pytest.fixture
def journal() -> EvolutionJournal:
    return EvolutionJournal()


@pytest.fixture
def state_machine(journal: EvolutionJournal) -> StateMachine:
    return StateMachine(journal=journal)


@pytest.fixture
def config() -> StemAgentConfig:
    return StemAgentConfig(openai_api_key="test-key-not-real")


@pytest.fixture
def registry() -> CapabilityRegistry:
    return build_default_registry()


@pytest.fixture
def small_corpus() -> list[BenchmarkSample]:
    """A small subset of the benchmark corpus for fast tests."""
    full = get_benchmark_corpus()
    # Pick one from each category + one clean sample
    return [full[0], full[5], full[10], full[12], full[15]]


@pytest.fixture
def tmp_dir() -> str:
    """Temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d
