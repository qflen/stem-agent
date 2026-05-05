"""Runtime executor for generated capability validators.

The capability_generation phase admits ``Capability`` objects whose
``validator_code`` is a sandboxed Python check function. Until the
dispatcher existed, that code only existed as a static-analysis curio:
the prompt fragment shaped the LLM, but the validator never ran. The
dispatcher closes the loop; at review time it executes every admitted
check on the code under review, with a per-check timeout and a
quarantine fence so a single misbehaving validator can't corrupt the
review pipeline.

Platform note: per-check timeouts use ``signal.SIGALRM``, which is
POSIX-only. macOS dev boxes and Linux CI work; Windows would need a
threading-based timeout the registry doesn't currently target.
"""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from stem_agent.capabilities.registry import Capability, CapabilityRegistry


@dataclass(frozen=True)
class GeneratedCheckFinding:
    """Result of running one admitted validator over the code under review."""

    name: str
    description: str
    hits: tuple[str, ...] = field(default_factory=tuple)


class _CheckTimeoutError(Exception):
    """Raised by the SIGALRM handler when a validator exceeds its budget."""


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _CheckTimeoutError("validator exceeded timeout")


@contextlib.contextmanager
def _bounded_alarm(seconds: float):
    """Set a SIGALRM-bounded budget and restore the previous handler on exit."""
    if seconds <= 0:
        yield
        return
    previous = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _compile_check(code: str) -> Callable[[str], list[str]]:
    namespace: dict[str, Any] = {}
    compiled = compile(code, "<generated_capability>", "exec")
    exec(compiled, namespace)
    candidate = namespace.get("check")
    if not callable(candidate):
        raise ValueError("validator_code must define a callable named 'check'")
    return candidate  # type: ignore[return-value]


class ReviewDispatcher:
    """Compiles and runs admitted validator_code at review time.

    Built once per agent (cheaper than recompiling on every review), the
    dispatcher caches the callable per capability name and runs each one
    against the code under review. A misbehaving validator (raises,
    returns the wrong type, or runs over budget) is quarantined for the
    rest of this dispatcher's life; the rest of the pipeline keeps
    working, and the journal will see the resulting reduction in
    findings.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        timeout_per_check: float = 0.5,
    ) -> None:
        self._registry = registry
        self._timeout = timeout_per_check
        self._compiled: dict[str, Callable[[str], list[str]]] = {}
        self._descriptions: dict[str, str] = {}
        self._quarantined: set[str] = set()
        self._compile_admitted()

    def _compile_admitted(self) -> None:
        for cap in self._registry.list_all():
            if cap.validator_code is None:
                continue
            try:
                self._compiled[cap.name] = _compile_check(cap.validator_code)
                self._descriptions[cap.name] = cap.description
            except (SyntaxError, ValueError, TypeError):
                self._quarantined.add(cap.name)

    @property
    def quarantined(self) -> frozenset[str]:
        return frozenset(self._quarantined)

    @property
    def admitted(self) -> tuple[str, ...]:
        return tuple(self._compiled.keys())

    def _run_one(self, name: str, check: Callable[[str], list[str]], code: str) -> list[str]:
        with _bounded_alarm(self._timeout):
            result = check(code)
        if not isinstance(result, list):
            raise TypeError("check must return list[str]")
        if not all(isinstance(item, str) for item in result):
            raise TypeError("check must return list[str]")
        return result

    def run(self, code: str) -> list[GeneratedCheckFinding]:
        findings: list[GeneratedCheckFinding] = []
        for name, check in list(self._compiled.items()):
            if name in self._quarantined:
                continue
            try:
                hits = self._run_one(name, check, code)
            except (_CheckTimeoutError, Exception):
                self._quarantined.add(name)
                continue
            if hits:
                findings.append(
                    GeneratedCheckFinding(
                        name=name,
                        description=self._descriptions.get(name, name),
                        hits=tuple(hits),
                    )
                )
        return findings


def format_dispatcher_findings(findings: list[GeneratedCheckFinding]) -> str:
    """Render dispatcher findings as a tool-block addendum the LLM reads."""
    if not findings:
        return "## Generated Check Findings\n- (no admitted validator fired)\n"
    lines = ["## Generated Check Findings"]
    for finding in findings:
        lines.append(f"- **{finding.name}**; {finding.description}")
        for hit in finding.hits:
            lines.append(f"    - {hit}")
    return "\n".join(lines) + "\n"


def maybe_make_dispatcher(
    registry: CapabilityRegistry,
    *,
    timeout_per_check: float = 0.5,
) -> ReviewDispatcher | None:
    """Return a dispatcher iff the registry has at least one validator to run."""
    if not any(c.validator_code for c in registry.list_all()):
        return None
    return ReviewDispatcher(registry, timeout_per_check=timeout_per_check)


def _capability_uses_runtime_check(cap: Capability) -> bool:
    return cap.validator_code is not None
