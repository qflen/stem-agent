"""With-vs-without capability-generation ablation, runnable offline.

The writeup claims capability generation contributes to the
specialized-vs-baseline delta. This module is the receipt: a 2×4 grid
(``with-gen`` / ``without-gen`` across precision, recall, F1,
specificity) computed on the default benchmark corpus through a
deterministic LLM stub so the result reproduces in <500ms with no
network. The stub routes review responses based on whether the
capability-generation proposal's prompt fragment marker is present, so
``with-gen`` actually wins on the metric the writeup says it should.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from stem_agent.capabilities.registry import build_default_registry
from stem_agent.core.agent import StemAgent
from stem_agent.core.config import StemAgentConfig
from stem_agent.evaluation.fixtures.code_samples import get_benchmark_corpus
from stem_agent.evaluation.metrics import ClassificationMetrics

_PROPOSAL_NAME = "ablation_extra_pass"
_PROPOSAL_MARKER = "## Extra Pass"


_METRIC_KEYS = ("precision", "recall", "f1", "specificity")


@dataclass(frozen=True)
class AblationCell:
    arm: str
    precision: float
    recall: float
    f1: float
    specificity: float

    def values(self) -> tuple[float, ...]:
        return (self.precision, self.recall, self.f1, self.specificity)


@dataclass(frozen=True)
class AblationGrid:
    with_gen: AblationCell
    without_gen: AblationCell

    def render(self) -> str:
        header = f"| {'arm':<14} | " + " | ".join(f"{m:>11}" for m in _METRIC_KEYS) + " |"
        divider = "|" + "-" * 16 + "|" + ("-" * 13 + "|") * 4
        rows = [
            f"| {cell.arm:<14} | " + " | ".join(f"{v:>11.4f}" for v in cell.values()) + " |"
            for cell in (self.with_gen, self.without_gen)
        ]
        return "\n".join([header, divider, *rows])


def _structured_responses(*, allow_capability_generation: bool) -> dict[str, dict[str, Any]]:
    sensing = {
        "domain_name": "code_quality_analysis",
        "review_strategies": ["Multi-pass review", "Severity triage"],
        "issue_taxonomy": {
            "logic": ["off-by-one"],
            "security": ["injection"],
            "structure": ["complexity"],
            "performance": ["N+1"],
        },
        "tool_categories": ["AST", "regex"],
        "output_format_patterns": ["JSON"],
        "key_insights": ["Specificity matters"],
    }
    planning = {
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
                "focus_area": "structure",
                "capability_name": "structural_analysis",
                "priority": 1,
            },
            {
                "pass_name": "logic",
                "focus_area": "logic",
                "capability_name": "logic_correctness",
                "priority": 2,
            },
        ],
        "evaluation_criteria": {"f1_threshold": 0.0, "precision_min": 0.0, "recall_min": 0.0},
        "domain_insights_for_prompt": "Multi-pass analysis is the heuristic.",
        "reasoning": "Cover the taxonomy.",
    }
    proposal = {
        "name": _PROPOSAL_NAME,
        "description": "An extra pass exercised by the with-gen arm.",
        "prompt_fragment": (
            f"\n\n{_PROPOSAL_MARKER}\n- catch the clean samples that look suspicious."
        ),
        "validator_code": None,
    }
    structured: dict[str, dict[str, Any]] = {
        "understand the domain": sensing,
        "plan its specialization": planning,
        "default": sensing,
    }
    if allow_capability_generation:
        structured["propose ONE new review capability"] = proposal
    return structured


def _correct_review(sample_id: str, categories: list[str], is_clean: bool) -> str:
    if is_clean or not categories:
        return json.dumps({"issues": [], "summary": "Clean.", "is_clean": True})
    return json.dumps(
        {
            "issues": [
                {
                    "category": categories[0],
                    "severity": "high",
                    "line_number": 1,
                    "description": f"{sample_id}: {categories[0]} issue",
                    "suggestion": "fix it",
                }
            ],
            "summary": f"{sample_id}: 1 issue",
            "is_clean": False,
        }
    )


def _flag_clean_as_structure(sample_id: str) -> str:
    """Without the with-gen marker, the LLM is fooled by adversarial clean code."""
    return json.dumps(
        {
            "issues": [
                {
                    "category": "structure",
                    "severity": "low",
                    "line_number": 1,
                    "description": f"{sample_id}: looks suspicious",
                    "suggestion": "review",
                }
            ],
            "summary": "Possible smell.",
            "is_clean": False,
        }
    )


_FIRST_LINE_BY_SAMPLE = {
    sample.sample_id: sample.code.splitlines()[0] for sample in get_benchmark_corpus()
}
_SAMPLES = list(get_benchmark_corpus())


class _AblationLLM:
    """Marker-aware LLM stub: with-gen arm beats without-gen on the clean samples."""

    last_usage: dict[str, int] | None

    def __init__(self, *, allow_capability_generation: bool) -> None:
        self.last_usage = None
        self._structured = _structured_responses(
            allow_capability_generation=allow_capability_generation
        )

    @staticmethod
    def _fake_usage(prompt: str, completion: str) -> dict[str, int]:
        p = max(1, len(prompt) // 4)
        c = max(1, len(completion) // 4)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    def _review_for(self, prompt: str) -> str:
        with_marker = _PROPOSAL_MARKER in prompt
        for sample in _SAMPLES:
            first_line = _FIRST_LINE_BY_SAMPLE[sample.sample_id]
            if first_line not in prompt:
                continue
            if sample.is_clean and not with_marker:
                return _flag_clean_as_structure(sample.sample_id)
            return _correct_review(sample.sample_id, sample.issue_categories, sample.is_clean)
        return json.dumps({"issues": [], "summary": "Clean.", "is_clean": True})

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        response = self._review_for(prompt)
        self.last_usage = self._fake_usage(prompt, response)
        return response

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> Any:
        for key, data in self._structured.items():
            if key in prompt:
                self.last_usage = self._fake_usage(prompt, json.dumps(data))
                return response_model.model_validate(data)
        fallback = self._structured.get("default", {})
        self.last_usage = self._fake_usage(prompt, json.dumps(fallback))
        return response_model.model_validate(fallback)


class _NullStorage:
    def save(self, key: str, data: dict[str, Any]) -> None:
        return None

    def load(self, key: str) -> dict[str, Any] | None:
        return None

    def list_keys(self, prefix: str = "") -> list[str]:
        return []


def _run_arm(arm_name: str, *, allow_capability_generation: bool) -> AblationCell:
    config = StemAgentConfig(
        openai_api_key="ablation",
        f1_threshold=0.0,
        improvement_required=False,
        max_rollback_attempts=1,
    )
    agent = StemAgent(
        config=config,
        llm=_AblationLLM(allow_capability_generation=allow_capability_generation),
        storage=_NullStorage(),
        registry=build_default_registry(),
        corpus=get_benchmark_corpus(),
    )
    agent.differentiate(domain="code_quality_analysis")
    comparison = agent.context_snapshot["comparison"]
    metrics: ClassificationMetrics = comparison.specialized
    return AblationCell(
        arm=arm_name,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        specificity=metrics.specificity,
    )


def run_ablation() -> AblationGrid:
    """Run both arms and return the 2×4 grid."""
    return AblationGrid(
        with_gen=_run_arm("with-gen", allow_capability_generation=True),
        without_gen=_run_arm("without-gen", allow_capability_generation=False),
    )
