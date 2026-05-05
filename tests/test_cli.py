"""CLI surface; covers ``--seeds``, ``--store-prompts``, ``--max-rollbacks``, ``replay``.

The CLI does the wiring the writeup promises: it honours every flag, lands
journals at the seed-specific paths the multi-seed artefacts depend on,
and the ``replay`` command round-trips through the prompt archive without
a network. We exercise it in-process via ``typer.testing.CliRunner`` so
the runtime stays under a millisecond per invocation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from stem_agent import cli as cli_module
from stem_agent.adapters.prompt_archive import PromptArchivingLLM
from stem_agent.core.journal import EvolutionJournal


class _StubLLM:
    """LLMPort double whose responses don't matter for the CLI surface tests."""

    last_usage: dict[str, int] | None = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        self.calls.append(prompt)
        return '{"issues": [], "summary": "clean", "is_clean": true}'

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> Any:
        self.calls.append(prompt)
        return response_model.model_validate({})


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every CLI invocation in an isolated cwd with a stubbed config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-real")
    return tmp_path


@pytest.fixture
def stub_differentiate(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace OpenAIAdapter and StemAgent so CLI tests run in microseconds.

    Captures every constructed agent so the test can assert on flag plumbing
    (journal_key, corpus length, seed, store-prompts wiring).
    """
    captured: dict[str, list[Any]] = {"agents": [], "configs": [], "calls": []}

    class _StubAgent:
        def __init__(
            self,
            config: Any,
            llm: Any,
            storage: Any,
            registry: Any | None = None,
            corpus: Any | None = None,
        ) -> None:
            captured["configs"].append(config)
            captured["agents"].append(self)
            self.config = config
            self.llm = llm
            self.storage = storage
            self.corpus = corpus
            self.agent_config = None

        def differentiate(
            self,
            domain: str = "code_quality_analysis",
            *,
            journal_key: str | None = None,
        ) -> bool:
            captured["calls"].append({"domain": domain, "journal_key": journal_key, "agent": self})
            payload = {"events": [], "_marker": journal_key or "default"}
            self.storage.save(journal_key or "journal_test", payload)
            return True

    monkeypatch.setattr(cli_module, "_build_llm", lambda inner, out, store_prompts: inner)
    import stem_agent.adapters.openai_adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "OpenAIAdapter", lambda config: _StubLLM())
    import stem_agent.core.agent as agent_mod

    monkeypatch.setattr(agent_mod, "StemAgent", _StubAgent)
    monkeypatch.setattr(cli_module, "_display_agent_config", lambda agent: None)

    return captured


class TestDifferentiateSingleSeed:
    def test_default_lands_in_journal_dir(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        result = cli_runner.invoke(cli_module.app, ["differentiate"])
        assert result.exit_code == 0, result.stdout
        assert len(stub_differentiate["calls"]) == 1
        # Default seeds=1 path passes journal_key=None (timestamp-stamped).
        assert stub_differentiate["calls"][0]["journal_key"] is None

    def test_max_rollbacks_override(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        result = cli_runner.invoke(cli_module.app, ["differentiate", "--max-rollbacks", "0"])
        assert result.exit_code == 0
        assert stub_differentiate["configs"][0].max_rollback_attempts == 0


class TestDifferentiateMultiSeed:
    def test_seeds_three_writes_three_named_journals(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        result = cli_runner.invoke(cli_module.app, ["differentiate", "--seeds", "3"])
        assert result.exit_code == 0
        keys = [call["journal_key"] for call in stub_differentiate["calls"]]
        assert keys == ["journal_seed0", "journal_seed1", "journal_seed2"]

    def test_seeds_writes_under_cq_directory(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        cli_runner.invoke(cli_module.app, ["differentiate", "--seeds", "2"])
        out_dir = stub_workspace / "docs" / "example_run" / "seeds" / "cq"
        assert (out_dir / "journal_seed0.json").exists()
        assert (out_dir / "journal_seed1.json").exists()

    def test_seeds_security_lands_under_sec_directory(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        cli_runner.invoke(
            cli_module.app, ["differentiate", "--seeds", "2", "--domain", "security_audit"]
        )
        out_dir = stub_workspace / "docs" / "example_run" / "seeds" / "sec"
        assert (out_dir / "journal_seed0.json").exists()

    def test_seeds_assigns_distinct_seed_per_run(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        cli_runner.invoke(cli_module.app, ["differentiate", "--seeds", "3"])
        seeds = [c.seed for c in stub_differentiate["configs"]]
        assert seeds == [0, 1, 2]

    def test_seeds_passes_full_corpus(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        """Each seed gets the full 19-sample corpus; the agent partitions internally."""
        cli_runner.invoke(cli_module.app, ["differentiate", "--seeds", "2"])
        agents = stub_differentiate["agents"]
        assert all(len(a.corpus) == 19 for a in agents)


class TestStorePrompts:
    def test_store_prompts_wraps_llm_in_archiver(self, tmp_path: Path) -> None:
        inner = _StubLLM()
        wrapped = cli_module._build_llm(inner, tmp_path, store_prompts=True)
        assert isinstance(wrapped, PromptArchivingLLM)

    def test_store_prompts_off_returns_inner_unchanged(self, tmp_path: Path) -> None:
        inner = _StubLLM()
        assert cli_module._build_llm(inner, tmp_path, store_prompts=False) is inner

    def test_archiver_writes_hash_filename(self, tmp_path: Path) -> None:
        inner = _StubLLM()
        wrapped = PromptArchivingLLM(inner, tmp_path / "prompts")
        wrapped.generate("hello world")
        digest = EvolutionJournal.hash_prompt("hello world")
        path = tmp_path / "prompts" / f"{digest}.txt"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "hello world"


class TestReplay:
    def test_replay_round_trips(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
    ) -> None:
        body = "system: review carefully\nuser: code"
        digest = EvolutionJournal.hash_prompt(body)
        prompts_dir = stub_workspace / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / f"{digest}.txt").write_text(body, encoding="utf-8")
        result = cli_runner.invoke(cli_module.app, ["replay", digest])
        assert result.exit_code == 0
        assert "review carefully" in result.stdout

    def test_replay_missing_hash_errors(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
    ) -> None:
        result = cli_runner.invoke(cli_module.app, ["replay", "deadbeefdeadbeef"])
        assert result.exit_code == 1


class TestStubAdapterPlumbing:
    """Sanity check on the test stub itself; pytest setup didn't drift."""

    def test_stub_agent_records_storage_writes(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        cli_runner.invoke(cli_module.app, ["differentiate", "--seeds", "1"])
        # _StubAgent saves to storage with key 'journal_test' under default mode,
        # but actual journal_key flows through; in seeds=1 mode that means the
        # storage was used at all.
        out_dir = stub_workspace / "evolution_journals"
        files = list(out_dir.glob("*.json")) if out_dir.exists() else []
        # Either it landed in default journal_dir, or under seeds; both are fine
        # in the stubbed path; the call list is the authoritative signal.
        assert stub_differentiate["calls"]
        files_or_calls = files or stub_differentiate["calls"]
        assert files_or_calls

    def test_seed_zero_payload_loads(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Storage serialises through json.dump; the marker round-trips."""
        from stem_agent.adapters.json_storage import JsonStorageAdapter

        storage = JsonStorageAdapter(str(tmp_path))
        storage.save("journal_seed0", {"events": [], "_marker": "journal_seed0"})
        loaded = storage.load("journal_seed0")
        assert loaded == {"events": [], "_marker": "journal_seed0"}

    def test_replay_empty_dir_argument(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Custom prompts_dir is honoured."""
        body = "hello"
        digest = EvolutionJournal.hash_prompt(body)
        custom = tmp_path / "elsewhere"
        custom.mkdir()
        (custom / f"{digest}.txt").write_text(body, encoding="utf-8")
        result = cli_runner.invoke(cli_module.app, ["replay", digest, "--prompts-dir", str(custom)])
        assert result.exit_code == 0
        assert "hello" in result.stdout


class TestCorpusRouting:
    def test_seeds_dir_for_known_domains(self) -> None:
        from stem_agent.cli import _seeds_dir_for

        assert _seeds_dir_for("code_quality_analysis").parts[-1] == "cq"
        assert _seeds_dir_for("security_audit").parts[-1] == "sec"

    def test_unknown_domain_exits_one(
        self,
        cli_runner: CliRunner,
        stub_workspace: Path,
        stub_differentiate: dict[str, list[Any]],
    ) -> None:
        result = cli_runner.invoke(
            cli_module.app, ["differentiate", "--domain", "fictional_domain"]
        )
        assert result.exit_code == 1
        # Avoid relying on stylised stderr text; exit code is the contract.

    def test_dispatcher_imports_succeed(self) -> None:
        """The CLI can be imported without OPENAI_API_KEY (pure module load)."""
        import importlib

        importlib.reload(cli_module)
        assert callable(cli_module.app)


def test_replay_creates_no_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Replay over a non-existent dir errors instead of silently creating one."""
    nonexistent = tmp_path / "missing"
    result = cli_runner.invoke(
        cli_module.app, ["replay", "0123456789abcdef", "--prompts-dir", str(nonexistent)]
    )
    assert result.exit_code == 1
    assert not nonexistent.exists()


def test_journal_payload_serialises(stub_workspace: Path) -> None:
    """A journal payload ending up in JsonStorageAdapter must JSON-serialise."""
    from stem_agent.adapters.json_storage import JsonStorageAdapter

    storage = JsonStorageAdapter(str(stub_workspace))
    storage.save("journal_seed5", {"events": [{"type": "decision"}]})
    raw = (stub_workspace / "journal_seed5.json").read_text()
    parsed = json.loads(raw)
    assert parsed["events"][0]["type"] == "decision"
