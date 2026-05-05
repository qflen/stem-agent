"""Cross-run capability priors derived from past journals.

Each saved journal records, in events, which capabilities the planning
phase selected and whether the run graduated to ``SPECIALIZED``. This
module aggregates those signals across every journal sharing the
current ``domain`` and exposes a per-capability success rate for the
planning phase to fold into its ranking. Laplace smoothing keeps the
weights well-defined when a capability is rarely (or never) selected,
and leaves capabilities with no history at the neutral 0.5; so a fresh
``evolution_journals/`` directory does not silently distort planning.
"""

from __future__ import annotations

from typing import Any

from stem_agent.ports.storage import StoragePort

NEUTRAL_WEIGHT = 0.5


def _domain_of(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("event_type") != "state_transition":
            continue
        ctx = event.get("data", {}).get("context", {})
        if isinstance(ctx, dict):
            domain = ctx.get("domain")
            if isinstance(domain, str) and domain:
                return domain
    return None


def _did_graduate(events: list[dict[str, Any]]) -> bool:
    for event in reversed(events):
        if event.get("event_type") != "state_transition":
            continue
        target = event.get("data", {}).get("to", "")
        return target in ("specialized", "executing")
    return False


def _selected_capabilities(events: list[dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    for event in events:
        if event.get("event_type") != "capability_added":
            continue
        data = event.get("data", {})
        reason = data.get("reason", "") or ""
        if not reason.startswith("Selected during planning"):
            continue
        name = data.get("capability")
        if isinstance(name, str) and name:
            selected.add(name)
    return selected


def weight_capabilities(domain: str, storage: StoragePort) -> dict[str, float]:
    """Return Laplace-smoothed graduation rate per capability for ``domain``.

    Walks every ``journal_*`` blob in ``storage``; counts how often each
    capability was selected during planning and how often the run
    graduated to ``SPECIALIZED``. Capabilities never selected do not
    appear in the result; the caller treats their absence as the neutral
    prior.
    """
    selected_total: dict[str, int] = {}
    succeeded: dict[str, int] = {}
    for key in storage.list_keys("journal_"):
        blob = storage.load(key)
        if not blob:
            continue
        events = blob.get("events", []) if isinstance(blob, dict) else []
        if not isinstance(events, list):
            continue
        if _domain_of(events) != domain:
            continue
        graduated = _did_graduate(events)
        for cap in _selected_capabilities(events):
            selected_total[cap] = selected_total.get(cap, 0) + 1
            if graduated:
                succeeded[cap] = succeeded.get(cap, 0) + 1
    return {cap: (succeeded.get(cap, 0) + 1) / (selected_total[cap] + 2) for cap in selected_total}


def weight_for(name: str, weights: dict[str, float]) -> float:
    """Look up ``name`` in ``weights``, defaulting to the neutral prior."""
    return weights.get(name, NEUTRAL_WEIGHT)
