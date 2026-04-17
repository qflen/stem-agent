"""Tests for static-tool invocation on the specialized review path.

The specialized agent used to *advertise* AST analysis and pattern
scanning in its prompt fragments without ever calling them — the tools
only ran in the cross-check, after the review was over. These tests
pin the wired-up behaviour: tools run before the LLM call, their
findings are injected into the prompt, and each invocation is recorded
in the journal alongside the LLM call it fed.
"""

from __future__ import annotations

from stem_agent.core.agent import StemAgent
from stem_agent.core.config import StemAgentConfig
from stem_agent.core.journal import EventType, EvolutionJournal
from stem_agent.evaluation.benchmark import format_tool_findings, make_llm_review_fn
from stem_agent.evaluation.fixtures.code_samples import get_benchmark_corpus
from tests.conftest import FakeLLM


class TestFormatToolFindings:
    """format_tool_findings renders a compact block the LLM can read."""

    def test_parse_failure_surface(self) -> None:
        block = format_tool_findings(None, [])
        assert "did not parse" in block

    def test_pattern_match_included(self) -> None:
        from stem_agent.capabilities.tools import analyze_structure, scan_patterns

        code = "def handler(x):\n    result = eval(x)\n    return result\n"
        metrics = analyze_structure(code)
        patterns = scan_patterns(code)
        block = format_tool_findings(metrics, patterns)
        assert "eval_or_exec=yes" in block
        assert "eval()" in block  # pattern description text
        assert "line 2" in block

    def test_no_patterns_says_so(self) -> None:
        from stem_agent.capabilities.tools import analyze_structure, scan_patterns

        code = "def add(a, b):\n    return a + b\n"
        block = format_tool_findings(analyze_structure(code), scan_patterns(code))
        assert "no known-unsafe patterns" in block


class TestMakeLlmReviewFnUseTools:
    """The review function runs the tools and records the run when asked."""

    def test_tools_block_included_in_prompt(self) -> None:
        captured: list[str] = []

        class _CapturingLLM:
            last_usage = None

            def generate(self, prompt: str, *, model: str | None = None) -> str:
                captured.append(prompt)
                return '{"issues": [], "summary": "clean", "is_clean": true}'

            def structured_generate(self, *_args, **_kw):  # noqa: D401
                raise AssertionError("not expected here")

        journal = EvolutionJournal()
        fn = make_llm_review_fn(_CapturingLLM(), journal=journal, use_tools=True)
        code = "def unsafe(x):\n    return eval(x)\n"
        fn(code, "SYSTEM")

        assert len(captured) == 1
        prompt = captured[0]
        assert "## Static Tool Findings" in prompt
        assert "eval_or_exec=yes" in prompt
        assert "## Code to Review" in prompt
        assert prompt.index("Static Tool Findings") < prompt.index("## Code to Review")

        decisions = journal.get_events_by_type(EventType.DECISION)
        assert any(
            "analyze_structure" in d.data["decision"] and "scan_patterns" in d.data["decision"]
            for d in decisions
        )

    def test_tools_skipped_when_flag_off(self) -> None:
        captured: list[str] = []

        class _CapturingLLM:
            last_usage = None

            def generate(self, prompt: str, *, model: str | None = None) -> str:
                captured.append(prompt)
                return '{"issues": [], "summary": "clean", "is_clean": true}'

            def structured_generate(self, *_args, **_kw):  # noqa: D401
                raise AssertionError("not expected here")

        journal = EvolutionJournal()
        fn = make_llm_review_fn(_CapturingLLM(), journal=journal, use_tools=False)
        fn("def f():\n    return eval('1')\n", "SYSTEM")

        assert "## Static Tool Findings" not in captured[0]
        tool_decisions = [
            d
            for d in journal.get_events_by_type(EventType.DECISION)
            if "analyze_structure" in d.data["decision"]
        ]
        assert not tool_decisions


class TestAgentReviewInvokesTools:
    """End-to-end: StemAgent.review runs the tools too."""

    def test_review_injects_tool_findings_and_logs(self, fake_llm: FakeLLM) -> None:
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )
        agent = StemAgent(config=config, llm=fake_llm, corpus=get_benchmark_corpus())
        agent.differentiate(domain="code_quality_analysis")

        agent.review("def handler(x):\n    return eval(x)\n")

        review_decisions = [
            d
            for d in agent.journal.get_events_by_type(EventType.DECISION)
            if d.phase == "review" and "analyze_structure" in d.data["decision"]
        ]
        assert review_decisions, "review() must log a tool-invocation DECISION"

        # The prompt the LLM saw should carry the findings block.
        generate_calls = [c for c in fake_llm.calls if c["method"] == "generate"]
        last_prompt = generate_calls[-1]["prompt"]
        assert "## Static Tool Findings" in last_prompt
        assert "eval_or_exec=yes" in last_prompt

    def test_validation_specialized_pass_logs_tool_events(self, fake_llm: FakeLLM) -> None:
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )
        agent = StemAgent(config=config, llm=fake_llm, corpus=get_benchmark_corpus())
        agent.differentiate(domain="code_quality_analysis")

        # Every specialized review during validation should have logged a
        # tool invocation alongside its LLM call.
        tool_events = [
            d
            for d in agent.journal.get_events_by_type(EventType.DECISION)
            if d.phase == "validation_specialized_tools"
        ]
        assert len(tool_events) == 20, (
            f"expected one tool-invocation DECISION per benchmark sample, got {len(tool_events)}"
        )
