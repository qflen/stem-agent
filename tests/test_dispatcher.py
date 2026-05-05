"""ReviewDispatcher; runs admitted validators with a quarantine fence.

Validates: compilation caching, timeout enforcement, quarantine on
exception/wrong-type/timeout, and integration with ``make_llm_review_fn``
so the dispatcher's findings reach the prompt only on the specialized arm.
"""

from __future__ import annotations

import sys

import pytest

from stem_agent.capabilities.dispatcher import (
    GeneratedCheckFinding,
    ReviewDispatcher,
    format_dispatcher_findings,
    maybe_make_dispatcher,
)
from stem_agent.capabilities.registry import (
    Capability,
    CapabilityCategory,
    CapabilityRegistry,
)
from stem_agent.core.journal import EventType, EvolutionJournal
from stem_agent.evaluation.benchmark import make_llm_review_fn

if sys.platform == "win32":  # pragma: no cover - dispatcher uses SIGALRM
    pytest.skip("ReviewDispatcher requires POSIX SIGALRM", allow_module_level=True)


def _registry_with(*caps: Capability) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for cap in caps:
        registry.register(cap)
    return registry


def _generated(name: str, code: str, description: str = "test cap") -> Capability:
    return Capability(
        name=name,
        category=CapabilityCategory.DETECTION,
        description=description,
        prompt_fragment="## Test\n- nothing.",
        origin="generated",
        validator_code=code,
    )


VALID_CHECK = """
import re
def check(code):
    return ['eval'] if re.search(r'\\beval\\(', code) else []
"""

EXCEPTION_CHECK = """
def check(code):
    raise RuntimeError('boom')
"""

WRONG_TYPE_CHECK = """
def check(code):
    return 'not a list'
"""

NON_LIST_ELEMENT_CHECK = """
def check(code):
    return [1, 2, 3]
"""

INFINITE_LOOP_CHECK = """
def check(code):
    while True:
        pass
    return []
"""


class TestCompilation:
    def test_admitted_compiles_once(self) -> None:
        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = ReviewDispatcher(registry)
        first = dispatcher._compiled["eval_check"]
        again = ReviewDispatcher(registry)._compiled["eval_check"]
        assert first is not again  # different instances per dispatcher
        # Within one instance the cached callable is stable across run() calls.
        dispatcher.run("eval(x)")
        assert dispatcher._compiled["eval_check"] is first

    def test_capability_without_validator_is_skipped(self) -> None:
        registry = _registry_with(
            Capability(
                name="prompt_only",
                category=CapabilityCategory.DETECTION,
                description="no static check",
                prompt_fragment="## Prompt-only\n",
                origin="generated",
                validator_code=None,
            ),
            _generated("eval_check", VALID_CHECK),
        )
        dispatcher = ReviewDispatcher(registry)
        assert dispatcher.admitted == ("eval_check",)


class TestRunFindings:
    def test_finding_returned_when_check_fires(self) -> None:
        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = ReviewDispatcher(registry)
        findings = dispatcher.run("def x(): return eval('1')\n")
        assert findings == [
            GeneratedCheckFinding(name="eval_check", description="test cap", hits=("eval",))
        ]

    def test_no_finding_returned_when_check_silent(self) -> None:
        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = ReviewDispatcher(registry)
        assert dispatcher.run("def x(): return 1") == []


class TestQuarantine:
    def test_exception_quarantines_check(self) -> None:
        registry = _registry_with(_generated("explodes", EXCEPTION_CHECK))
        dispatcher = ReviewDispatcher(registry)
        assert dispatcher.run("anything") == []
        assert "explodes" in dispatcher.quarantined
        # Subsequent runs skip the quarantined check entirely
        assert dispatcher.run("again") == []

    def test_wrong_return_type_quarantines(self) -> None:
        registry = _registry_with(_generated("typeo", WRONG_TYPE_CHECK))
        dispatcher = ReviewDispatcher(registry)
        dispatcher.run("anything")
        assert "typeo" in dispatcher.quarantined

    def test_non_string_element_quarantines(self) -> None:
        registry = _registry_with(_generated("ints", NON_LIST_ELEMENT_CHECK))
        dispatcher = ReviewDispatcher(registry)
        dispatcher.run("anything")
        assert "ints" in dispatcher.quarantined

    def test_timeout_quarantines(self) -> None:
        registry = _registry_with(_generated("slow", INFINITE_LOOP_CHECK))
        dispatcher = ReviewDispatcher(registry, timeout_per_check=0.05)
        dispatcher.run("anything")
        assert "slow" in dispatcher.quarantined

    def test_compile_error_quarantines_at_init(self) -> None:
        registry = _registry_with(_generated("broken", "def check(:\n    return []\n"))
        dispatcher = ReviewDispatcher(registry)
        assert "broken" in dispatcher.quarantined
        assert dispatcher.run("anything") == []


class TestMaybeMakeDispatcher:
    def test_returns_none_when_no_validators(self) -> None:
        registry = CapabilityRegistry()
        registry.register(
            Capability(
                name="prompt_only",
                category=CapabilityCategory.DETECTION,
                description="no validator",
                prompt_fragment="## Prompt only\n",
                validator_code=None,
            )
        )
        assert maybe_make_dispatcher(registry) is None

    def test_returns_dispatcher_when_validator_present(self) -> None:
        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = maybe_make_dispatcher(registry)
        assert isinstance(dispatcher, ReviewDispatcher)


class TestFormatDispatcherFindings:
    def test_empty_findings_emit_placeholder(self) -> None:
        text = format_dispatcher_findings([])
        assert "Generated Check Findings" in text
        assert "no admitted validator fired" in text

    def test_findings_render_hits(self) -> None:
        finding = GeneratedCheckFinding(
            name="eval_check", description="catches eval", hits=("line 2", "line 4")
        )
        text = format_dispatcher_findings([finding])
        assert "eval_check" in text
        assert "catches eval" in text
        assert "line 2" in text
        assert "line 4" in text


class TestReviewFnIntegration:
    def test_dispatcher_findings_reach_prompt(self) -> None:
        captured: list[str] = []

        class _CapturingLLM:
            last_usage = None

            def generate(self, prompt: str, *, model: str | None = None) -> str:
                captured.append(prompt)
                return '{"issues": [], "summary": "clean", "is_clean": true}'

            def structured_generate(self, *_, **__):
                raise AssertionError("not exercised here")

        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = ReviewDispatcher(registry)
        journal = EvolutionJournal()
        review_fn = make_llm_review_fn(
            _CapturingLLM(), journal=journal, use_tools=True, dispatcher=dispatcher
        )
        review_fn("def x(): return eval('1')", "SYSTEM")

        assert "Generated Check Findings" in captured[0]
        assert "eval_check" in captured[0]
        assert "eval" in captured[0]

    def test_baseline_path_skips_dispatcher_even_when_supplied(self) -> None:
        captured: list[str] = []

        class _CapturingLLM:
            last_usage = None

            def generate(self, prompt: str, *, model: str | None = None) -> str:
                captured.append(prompt)
                return '{"issues": [], "summary": "clean", "is_clean": true}'

            def structured_generate(self, *_, **__):
                raise AssertionError("not exercised here")

        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = ReviewDispatcher(registry)
        review_fn = make_llm_review_fn(
            _CapturingLLM(), journal=None, use_tools=False, dispatcher=dispatcher
        )
        review_fn("def x(): return eval('1')", "SYSTEM")
        assert "Generated Check Findings" not in captured[0]
        assert "Static Tool Findings" not in captured[0]

    def test_dispatcher_count_logged_to_journal(self) -> None:
        class _StubLLM:
            last_usage = None

            def generate(self, prompt: str, *, model: str | None = None) -> str:
                return '{"issues": [], "summary": "clean", "is_clean": true}'

            def structured_generate(self, *_, **__):
                raise AssertionError("not exercised here")

        registry = _registry_with(_generated("eval_check", VALID_CHECK))
        dispatcher = ReviewDispatcher(registry)
        journal = EvolutionJournal()
        fn = make_llm_review_fn(_StubLLM(), journal=journal, use_tools=True, dispatcher=dispatcher)
        fn("def x(): return eval('1')", "SYSTEM")
        decisions = journal.get_events_by_type(EventType.DECISION)
        assert any("dispatcher_findings=1" in d.data["reasoning"] for d in decisions)
