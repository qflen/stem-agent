"""Tests for OpenAIAdapter retry, timeout, and usage handling.

Uses a flaky fake OpenAI client so we can exercise the RateLimitError
retry path without ever calling out to the real API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import openai
import pytest

from stem_agent.adapters.openai_adapter import OpenAIAdapter
from stem_agent.core.config import StemAgentConfig


@dataclass
class _FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


def _make_fake_response(content: str = '{"ok": true}') -> _FakeResponse:
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=content))],
        usage=_FakeUsage(),
    )


def _make_rate_limit_error() -> openai.RateLimitError:
    """Construct a RateLimitError without having to hit the API.

    The OpenAI SDK's RateLimitError signature requires a response object —
    we provide a minimal stand-in so the exception class matches.
    """

    class _Resp:
        status_code = 429
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.request = None

    return openai.RateLimitError("rate limited", response=_Resp(), body={"error": "rate_limit"})


class FlakyFakeOpenAIClient:
    """Raises RateLimitError on the first ``fail_first`` calls, then succeeds."""

    def __init__(self, fail_first: int = 2) -> None:
        self._fail_first = fail_first
        self.call_count = 0
        self.chat = self

    def create(self, **_kwargs: Any) -> _FakeResponse:  # openai.chat.completions.create
        self.call_count += 1
        if self.call_count <= self._fail_first:
            raise _make_rate_limit_error()
        return _make_fake_response()

    @property
    def completions(self) -> FlakyFakeOpenAIClient:
        return self


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch OpenAIAdapter._sleep so tests don't wait real seconds."""
    monkeypatch.setattr(OpenAIAdapter, "_sleep", lambda self, _seconds: None)


class TestRetryOnTransientErrors:
    def test_retries_then_succeeds_on_rate_limit(self) -> None:
        adapter = OpenAIAdapter(StemAgentConfig(openai_api_key="test"))
        adapter._client = FlakyFakeOpenAIClient(fail_first=2)

        result = adapter.generate("hello")

        assert result == '{"ok": true}'
        assert adapter._client.call_count == 3  # two failures + one success

    def test_captures_usage_after_successful_retry(self) -> None:
        adapter = OpenAIAdapter(StemAgentConfig(openai_api_key="test"))
        adapter._client = FlakyFakeOpenAIClient(fail_first=1)

        adapter.generate("hello")

        assert adapter.last_usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_raises_after_exhausting_retries(self) -> None:
        adapter = OpenAIAdapter(StemAgentConfig(openai_api_key="test"))
        adapter._client = FlakyFakeOpenAIClient(fail_first=99)  # always fails

        with pytest.raises(openai.RateLimitError):
            adapter.generate("hello")

        # Four attempts — the initial call plus three retries.
        assert adapter._client.call_count == 4

    def test_non_retryable_error_propagates_immediately(self) -> None:
        adapter = OpenAIAdapter(StemAgentConfig(openai_api_key="test"))

        class ExplodingClient:
            call_count = 0
            chat = None

            def __init__(self) -> None:
                self.chat = self

            @property
            def completions(self) -> ExplodingClient:
                return self

            def create(self, **_kwargs: Any) -> Any:
                self.call_count += 1
                raise ValueError("not retryable")

        adapter._client = ExplodingClient()
        with pytest.raises(ValueError, match="not retryable"):
            adapter.generate("hello")
        assert adapter._client.call_count == 1


class TestConfigWiring:
    def test_request_timeout_default(self) -> None:
        config = StemAgentConfig(openai_api_key="test")
        assert config.request_timeout == 60.0

    def test_request_timeout_overridable(self) -> None:
        config = StemAgentConfig(openai_api_key="test", request_timeout=10.0)
        assert config.request_timeout == 10.0
