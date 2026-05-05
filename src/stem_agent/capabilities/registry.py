"""Discoverable capability catalog.

Capabilities are named units of functionality that the agent can
select during differentiation. Each capability knows its purpose,
what it provides, and how to describe itself for inclusion in prompts.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CapabilityCategory(enum.Enum):
    """Categories of capabilities the agent can acquire."""

    ANALYSIS = "analysis"
    DETECTION = "detection"
    REPORTING = "reporting"
    TOOL = "tool"


@dataclass(frozen=True)
class Capability:
    """A named unit of functionality the agent can select.

    ``origin`` distinguishes hand-authored ``"registered"`` capabilities
    from ``"generated"`` ones proposed by the agent during the
    capability_generation phase. ``validator_code``, if set, is the
    sandboxed static-analysis helper the agent proposed alongside the
    prompt fragment; kept on the dataclass so the journal has a single
    artefact to point at.
    """

    name: str
    category: CapabilityCategory
    description: str
    prompt_fragment: str
    tags: frozenset[str] = field(default_factory=frozenset)
    origin: str = "registered"
    validator_code: str | None = None


class CapabilityRegistry:
    """Registry of all available capabilities.

    The sensing and planning phases query this registry to discover
    what the agent CAN become. Capabilities are registered at startup
    and are immutable thereafter.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        """Register a capability. Raises if the name is already taken."""
        if capability.name in self._capabilities:
            raise ValueError(f"Capability '{capability.name}' already registered")
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Capability | None:
        """Look up a capability by name."""
        return self._capabilities.get(name)

    def list_all(self) -> list[Capability]:
        """Return all registered capabilities."""
        return list(self._capabilities.values())

    def list_by_category(self, category: CapabilityCategory) -> list[Capability]:
        """Return capabilities in a given category."""
        return [c for c in self._capabilities.values() if c.category == category]

    def list_by_tag(self, tag: str) -> list[Capability]:
        """Return capabilities matching a tag."""
        return [c for c in self._capabilities.values() if tag in c.tags]

    def select(self, names: list[str]) -> list[Capability]:
        """Select multiple capabilities by name. Raises on unknown names."""
        result = []
        for name in names:
            cap = self._capabilities.get(name)
            if cap is None:
                raise KeyError(f"Unknown capability: '{name}'")
            result.append(cap)
        return result


def build_default_registry() -> CapabilityRegistry:
    """Build the default capability registry for code quality analysis."""
    registry = CapabilityRegistry()

    registry.register(
        Capability(
            name="structural_analysis",
            category=CapabilityCategory.ANALYSIS,
            description="Analyze code structure: function length, nesting, complexity",
            prompt_fragment=(
                "Analyze the structural properties of the code:\n"
                "- Function/method length (flag if >30 lines or >3 levels of nesting)\n"
                "- Cyclomatic complexity indicators (multiple branches, nested conditionals)\n"
                "- Dead code (unreachable statements after return/break/continue)\n"
                "- Unnecessary complexity (overly clever constructs, redundant operations)"
            ),
            tags=frozenset({"structure", "complexity", "maintainability"}),
        )
    )

    registry.register(
        Capability(
            name="logic_correctness",
            category=CapabilityCategory.DETECTION,
            description="Detect logic bugs: off-by-one, wrong operators, missing null checks",
            prompt_fragment=(
                "Check for logic correctness issues:\n"
                "- Off-by-one errors in loop bounds or slice indices\n"
                "- Wrong comparison operators (< vs <=, == vs is)\n"
                "- Missing None/null checks before attribute access\n"
                "- Integer overflow or underflow in arithmetic\n"
                "- Incorrect boolean logic (De Morgan violations, short-circuit errors)\n"
                "- Edge cases in boundary conditions"
            ),
            tags=frozenset({"logic", "bugs", "correctness"}),
        )
    )

    registry.register(
        Capability(
            name="security_analysis",
            category=CapabilityCategory.DETECTION,
            description="Identify security vulnerabilities: injection, traversal, credentials",
            prompt_fragment=(
                "Scan for security vulnerabilities:\n"
                "- SQL injection via string formatting or concatenation\n"
                "- Path traversal in file operations (unsanitized user paths)\n"
                "- Hardcoded credentials, API keys, or secrets\n"
                "- Use of eval/exec with unsanitized input\n"
                "- Command injection via os.system or subprocess with shell=True\n"
                "- Insecure deserialization (pickle with untrusted data)"
            ),
            tags=frozenset({"security", "vulnerabilities", "injection"}),
        )
    )

    registry.register(
        Capability(
            name="performance_analysis",
            category=CapabilityCategory.ANALYSIS,
            description="Spot performance issues: N+1 patterns, unnecessary copies",
            prompt_fragment=(
                "Look for performance issues:\n"
                "- N+1 query patterns (database calls inside loops)\n"
                "- Unnecessary data copying (list(), dict() on large structures in loops)\n"
                "- Algorithmic inefficiency (O(n²) when O(n) is possible)\n"
                "- Resource leaks (files/connections not properly closed)"
            ),
            tags=frozenset({"performance", "efficiency", "optimization"}),
        )
    )

    registry.register(
        Capability(
            name="style_consistency",
            category=CapabilityCategory.ANALYSIS,
            description="Check code style and maintainability: naming, magic numbers",
            prompt_fragment=(
                "Check code style and maintainability:\n"
                "- Magic numbers without named constants\n"
                "- Inconsistent naming conventions\n"
                "- Overly broad exception handling (bare except:)\n"
                "- Missing or misleading comments on complex logic"
            ),
            tags=frozenset({"style", "maintainability", "readability"}),
        )
    )

    registry.register(
        Capability(
            name="severity_ranking",
            category=CapabilityCategory.REPORTING,
            description="Rank issues by severity and provide actionable output",
            prompt_fragment=(
                "For each issue found, assign a severity:\n"
                "- CRITICAL: Security vulnerabilities, data loss risks, crashes\n"
                "- HIGH: Logic bugs that produce wrong results\n"
                "- MEDIUM: Code smells that impair maintainability\n"
                "- LOW: Style issues, minor inefficiencies\n"
                "Order issues by severity (critical first)."
            ),
            tags=frozenset({"reporting", "severity", "prioritization"}),
        )
    )

    return registry
