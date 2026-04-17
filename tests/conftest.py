"""Shared test fixtures."""

from __future__ import annotations

import pytest

from stem_agent.core.journal import EvolutionJournal
from stem_agent.core.state_machine import StateMachine


@pytest.fixture
def journal() -> EvolutionJournal:
    return EvolutionJournal()


@pytest.fixture
def state_machine(journal: EvolutionJournal) -> StateMachine:
    return StateMachine(journal=journal)
