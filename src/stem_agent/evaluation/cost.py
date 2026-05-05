"""Token-cost estimation + budget cap for the live-evaluation batch.

The batch is gated on a ``$25`` total cap. ``estimate_cost`` walks every
``LLM_CALL`` event in a journal and sums (prompt + completion) tokens
weighted by the per-model price table. ``cumulative_spend`` aggregates
across a directory of journals so the CLI can surface "we've already
spent $X" before kicking off another seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stem_agent.core.journal import EventType, EvolutionJournal

PRICE_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.010),
    "scripted-planner": (0.0, 0.0),
    "scripted-executor": (0.0, 0.0),
}

DEFAULT_BUDGET_CAP_USD = 25.0


class BudgetExceededError(Exception):
    """Raised when the cumulative spend exceeds the configured cap."""


@dataclass(frozen=True)
class CallCost:
    model: str
    tokens: int
    dollars: float


def _per_model_dollars(model: str, tokens: int) -> float:
    """Apportion total tokens to input/output at the SDK's reported ratio.

    The journal stores aggregate ``token_count``; without per-call splits
    we approximate as 60% input / 40% output, which matches the ratios the
    OpenAI SDK reports on chat-completion calls in practice.
    """
    rates = PRICE_PER_1K_TOKENS.get(model)
    if rates is None:
        raise KeyError(f"unknown model in price table: {model!r}")
    input_per_k, output_per_k = rates
    input_tokens = int(tokens * 0.6)
    output_tokens = tokens - input_tokens
    return (input_tokens / 1000) * input_per_k + (output_tokens / 1000) * output_per_k


def estimate_cost(journal: EvolutionJournal) -> float:
    """Return total dollars spent on LLM calls in ``journal``."""
    total = 0.0
    for event in journal.get_events_by_type(EventType.LLM_CALL):
        tokens = event.data.get("token_count")
        model = event.data.get("model") or "default"
        if not isinstance(tokens, int):
            continue
        total += _per_model_dollars(model, tokens)
    return total


def cumulative_spend(journal_dir: Path) -> float:
    """Sum ``estimate_cost`` across every ``journal_*.json`` in ``journal_dir``."""
    total = 0.0
    if not journal_dir.exists():
        return 0.0
    for path in sorted(journal_dir.glob("journal_*.json")):
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        journal = EvolutionJournal.from_dict(blob)
        total += estimate_cost(journal)
    return total


def assert_under_cap(running_total: float, cap: float = DEFAULT_BUDGET_CAP_USD) -> None:
    """Raise ``BudgetExceededError`` if running spend has crossed the cap."""
    if running_total > cap:
        raise BudgetExceededError(f"running total ${running_total:.2f} exceeds cap ${cap:.2f}")
