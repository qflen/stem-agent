"""Configuration management via pydantic-settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StemAgentConfig(BaseSettings):
    """Central configuration for the stem agent.

    Reads from environment variables and .env files. All thresholds
    and model parameters are configurable without code changes.
    """

    model_config = SettingsConfigDict(
        env_prefix="STEM_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    planning_model: str = Field(
        default="gpt-4o-mini",
        description="Model for sensing and planning phases (cheaper, faster)",
    )
    execution_model: str = Field(
        default="gpt-4o",
        description="Model for code review execution (more capable)",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for all differentiation calls (0 for reproducibility)",
    )
    request_timeout: float = Field(
        default=60.0,
        description="Per-request timeout in seconds passed to the OpenAI client",
    )
    seed: int = Field(
        default=0,
        description=(
            "Seed forwarded to the OpenAI sampler and used to derive the "
            "deterministic corpus partition. OpenAI treats seed as best-effort, "
            "not a hard reproducibility guarantee."
        ),
    )

    # Validation thresholds; guard predicates for state transitions
    f1_threshold: float = Field(
        default=0.6,
        description="Minimum F1 score for VALIDATING → SPECIALIZED transition",
    )
    improvement_required: bool = Field(
        default=True,
        description="Whether specialized F1 must exceed baseline F1",
    )
    max_rollback_attempts: int = Field(
        default=3,
        description="Maximum VALIDATING → ROLLBACK cycles before graceful failure",
    )
    token_budget_cap: int | None = Field(
        default=None,
        description=(
            "Optional ceiling on cumulative tokens; when set, blocks graduation if "
            "the journal's total_tokens has already exceeded the cap. None disables."
        ),
    )

    # Storage
    journal_dir: str = Field(
        default="./evolution_journals",
        description="Directory for persisting evolution journals",
    )
    state_dir: str = Field(
        default="./agent_states",
        description="Directory for persisting agent state snapshots",
    )
