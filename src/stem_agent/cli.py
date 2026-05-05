"""CLI interface for the stem agent; powered by typer + rich."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from stem_agent.adapters.json_storage import JsonStorageAdapter
from stem_agent.adapters.prompt_archive import PromptArchivingLLM
from stem_agent.core.config import StemAgentConfig
from stem_agent.core.journal import EventType, EvolutionJournal
from stem_agent.evaluation.cost import DEFAULT_BUDGET_CAP_USD, cumulative_spend
from stem_agent.evaluation.fixtures.code_samples import (
    get_benchmark_corpus,
)
from stem_agent.evaluation.fixtures.security_audit_samples import (
    get_security_audit_corpus,
)
from stem_agent.ports.llm import LLMPort

app = typer.Typer(
    name="stem-agent",
    help="A self-specializing AI agent that differentiates for a task domain.",
    no_args_is_help=True,
)
console = Console()


_DOMAIN_CORPUS = {
    "code_quality_analysis": get_benchmark_corpus,
    "security_audit": get_security_audit_corpus,
}


def _seeds_dir_for(domain: str) -> Path:
    """Return the per-domain artifact directory under ``docs/example_run/seeds``."""
    short = "cq" if domain == "code_quality_analysis" else "sec"
    return Path("docs") / "example_run" / "seeds" / short


@app.command()
def differentiate(
    domain: str = typer.Option(
        "code_quality_analysis",
        "--domain",
        "-d",
        help="Domain signal to specialize for",
    ),
    seeds: int = typer.Option(
        1,
        "--seeds",
        "-n",
        min=1,
        help=(
            "Run differentiate this many times with seeds 0..N-1; each seed picks "
            "a deterministic corpus partition and is forwarded to the LLM sampler."
        ),
    ),
    store_prompts: bool = typer.Option(
        False,
        "--store-prompts",
        help="Write every composed prompt to <out>/prompts/<sha256_short>.txt.",
    ),
    max_rollbacks: int | None = typer.Option(
        None,
        "--max-rollbacks",
        help="Override max_rollback_attempts for this run.",
    ),
) -> None:
    """Run the full differentiation process: sense → plan → specialize → validate."""
    base_config = StemAgentConfig()

    if not base_config.openai_api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY not set. "
            "Set it via environment variable or .env file."
        )
        raise typer.Exit(1)

    if domain not in _DOMAIN_CORPUS:
        console.print(f"[bold red]Unknown domain:[/bold red] {domain}")
        raise typer.Exit(1)

    from stem_agent.adapters.openai_adapter import OpenAIAdapter
    from stem_agent.core.agent import StemAgent

    base_corpus = _DOMAIN_CORPUS[domain]()

    if seeds == 1:
        out_dir = Path(base_config.journal_dir)
        config = base_config.model_copy(
            update=_run_overrides(seed=base_config.seed, max_rollbacks=max_rollbacks),
        )
        llm = _build_llm(OpenAIAdapter(config), out_dir, store_prompts=store_prompts)
        storage = JsonStorageAdapter(str(out_dir))
        agent = StemAgent(config=config, llm=llm, storage=storage, corpus=base_corpus)
        success = agent.differentiate(domain=domain)
        if success:
            _display_agent_config(agent)
        else:
            console.print(
                "[bold red]Differentiation failed after max rollback attempts.[/bold red]"
            )
            raise typer.Exit(1)
        return

    out_dir = _seeds_dir_for(domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[int] = []
    for i in range(seeds):
        running = cumulative_spend(out_dir)
        if running >= DEFAULT_BUDGET_CAP_USD:
            console.print(
                f"[bold red]Budget cap ${DEFAULT_BUDGET_CAP_USD:.2f} reached "
                f"(spent ${running:.2f}); aborting before seed {i}.[/bold red]"
            )
            raise typer.Exit(1)
        console.print(
            f"\n[bold cyan]── Seed {i} ({domain}, "
            f"running ${running:.2f}/${DEFAULT_BUDGET_CAP_USD:.0f}) ──[/bold cyan]"
        )
        seed_config = base_config.model_copy(
            update=_run_overrides(seed=i, max_rollbacks=max_rollbacks),
        )
        llm = _build_llm(OpenAIAdapter(seed_config), out_dir, store_prompts=store_prompts)
        storage = JsonStorageAdapter(str(out_dir))
        # The agent partitions the full corpus by config.seed; the validation
        # phase runs on partition.validation, the empirical-holdout gate uses
        # partition.holdout, and sensing embeds partition.probe.
        agent = StemAgent(config=seed_config, llm=llm, storage=storage, corpus=base_corpus)
        success = agent.differentiate(domain=domain, journal_key=f"journal_seed{i}")
        if not success:
            failed.append(i)

    if failed:
        console.print(f"[bold red]{len(failed)}/{seeds} seeds failed: {failed}[/bold red]")
        raise typer.Exit(1)


def _run_overrides(*, seed: int, max_rollbacks: int | None) -> dict[str, object]:
    overrides: dict[str, object] = {"seed": seed}
    if max_rollbacks is not None:
        overrides["max_rollback_attempts"] = max_rollbacks
    return overrides


def _build_llm(inner: LLMPort, out_dir: Path, *, store_prompts: bool) -> LLMPort:
    if not store_prompts:
        return inner
    return PromptArchivingLLM(inner, out_dir / "prompts")


@app.command(name="eval-ablation")
def eval_ablation() -> None:
    """Print the with-vs-without capability-generation ablation grid."""
    from stem_agent.evaluation.ablation import run_ablation

    grid = run_ablation()
    console.print(grid.render())


@app.command()
def report(
    seeds_dir: str = typer.Argument(help="Directory containing journal_seedN.json files."),
    n_resamples: int = typer.Option(1000, "--resamples", "-r", help="Bootstrap resamples."),
) -> None:
    """Print the headline ``mean [lo, hi]`` table from a multi-seed run."""
    from stem_agent.evaluation.bootstrap import headline_table

    path = Path(seeds_dir)
    if not path.exists():
        console.print(f"[bold red]Seeds dir not found:[/bold red] {seeds_dir}")
        raise typer.Exit(1)
    console.print(headline_table(path, n_resamples=n_resamples))


@app.command()
def replay(
    prompt_hash: str = typer.Argument(help="16-character prompt hash from a journal LLM_CALL."),
    prompts_dir: str = typer.Option(
        "prompts",
        "--prompts-dir",
        "-d",
        help="Directory containing the archived prompt bodies.",
    ),
) -> None:
    """Print the body of a stored prompt by its journal hash."""
    path = Path(prompts_dir) / f"{prompt_hash}.txt"
    if not path.exists():
        console.print(f"[bold red]Prompt not found:[/bold red] {path}")
        raise typer.Exit(1)
    console.print(path.read_text(encoding="utf-8"))


@app.command()
def evaluate(
    journal_path: str = typer.Option(
        "",
        "--journal-path",
        "-j",
        help="Path to a specific journal JSON file. If empty, uses the latest.",
    ),
) -> None:
    """Display evaluation results from the latest (or specified) journal."""
    config = StemAgentConfig()
    storage = JsonStorageAdapter(config.journal_dir)

    if journal_path:
        data = json.loads(Path(journal_path).read_text())
    else:
        keys = storage.list_keys("journal_")
        if not keys:
            console.print("[yellow]No journals found. Run 'differentiate' first.[/yellow]")
            raise typer.Exit(1)
        data = storage.load(keys[-1])
        if data is None:
            console.print("[yellow]Journal data is empty.[/yellow]")
            raise typer.Exit(1)
        console.print(f"[dim]Loading journal: {keys[-1]}[/dim]\n")

    journal = EvolutionJournal.from_dict(data)
    _display_evaluation(journal)


@app.command()
def review(
    file_path: str = typer.Argument(help="Path to the Python file to review"),
) -> None:
    """Review a Python file using the specialized agent."""
    config = StemAgentConfig()

    if not config.openai_api_key:
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set.")
        raise typer.Exit(1)

    path = Path(file_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(1)

    code = path.read_text()

    from stem_agent.adapters.openai_adapter import OpenAIAdapter
    from stem_agent.core.agent import StemAgent

    llm = OpenAIAdapter(config)
    storage = JsonStorageAdapter(config.journal_dir)
    agent = StemAgent(config=config, llm=llm, storage=storage)

    console.print("[dim]Differentiating agent for code review...[/dim]\n")
    success = agent.differentiate(domain="code_quality_analysis")

    if not success:
        console.print("[bold red]Failed to specialize agent.[/bold red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Reviewing: {file_path}[/bold]\n")
    result = agent.review(code)
    _display_review_result(result)


@app.command(name="journal")
def show_journal(
    last: bool = typer.Option(False, "--last", "-l", help="Show the latest journal"),
    path: str = typer.Option("", "--path", "-p", help="Path to a specific journal file"),
) -> None:
    """Pretty-print an evolution journal."""
    config = StemAgentConfig()

    if path:
        data = json.loads(Path(path).read_text())
    elif last:
        storage = JsonStorageAdapter(config.journal_dir)
        keys = storage.list_keys("journal_")
        if not keys:
            console.print("[yellow]No journals found.[/yellow]")
            raise typer.Exit(1)
        data = storage.load(keys[-1])
        if data is None:
            console.print("[yellow]Journal data is empty.[/yellow]")
            raise typer.Exit(1)
        console.print(f"[dim]Journal: {keys[-1]}[/dim]\n")
    else:
        console.print("[yellow]Specify --last or --path.[/yellow]")
        raise typer.Exit(1)

    journal = EvolutionJournal.from_dict(data)
    _display_journal(journal)


def _display_agent_config(agent: object) -> None:
    """Display the specialized agent's configuration."""
    from stem_agent.core.agent import StemAgent

    ag: StemAgent = agent
    config = ag.agent_config
    if config is None:
        return

    table = Table(title="Specialized Agent Configuration")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Model", config.model)
    table.add_row("Capabilities", ", ".join(config.capabilities))
    table.add_row("Review Passes", ", ".join(config.review_passes))
    table.add_row("Prompt Length", f"{len(config.system_prompt)} chars")
    console.print(table)


def _display_evaluation(journal: EvolutionJournal) -> None:
    """Display evaluation metrics from journal events."""
    metric_events = journal.get_events_by_type(EventType.METRIC_MEASUREMENT)

    if not metric_events:
        console.print("[yellow]No metric events found in journal.[/yellow]")
        return

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Specialized", justify="right")
    table.add_column("Delta", justify="right")

    baseline: dict[str, float] = {}
    specialized: dict[str, float] = {}

    for event in metric_events:
        for key, value in event.data.items():
            if key.startswith("baseline_"):
                metric_name = key.replace("baseline_", "")
                baseline[metric_name] = value
            elif key.startswith("specialized_"):
                metric_name = key.replace("specialized_", "")
                specialized[metric_name] = value

    for metric in ["precision", "recall", "f1", "specificity"]:
        b = baseline.get(metric, 0.0)
        s = specialized.get(metric, 0.0)
        delta = s - b
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
        table.add_row(
            metric.capitalize(),
            f"{b:.4f}",
            f"{s:.4f}",
            f"[{delta_style}]{delta:+.4f}[/{delta_style}]",
        )

    console.print(table)


def _display_review_result(result: dict) -> None:
    """Display a structured review result."""
    issues = result.get("issues", [])

    if not issues:
        console.print(Panel("[green]No issues found; code looks clean.[/green]"))
        return

    table = Table(title=f"Review Results ({len(issues)} issues)")
    table.add_column("Severity", style="bold")
    table.add_column("Category")
    table.add_column("Line")
    table.add_column("Description")
    table.add_column("Suggestion")

    severity_styles = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "dim",
    }

    for issue in issues:
        sev = issue.get("severity", "medium")
        style = severity_styles.get(sev, "")
        table.add_row(
            f"[{style}]{sev.upper()}[/{style}]",
            issue.get("category", ""),
            str(issue.get("line_number", "")),
            issue.get("description", ""),
            issue.get("suggestion", ""),
        )

    console.print(table)

    summary = result.get("summary", "")
    if summary:
        console.print(f"\n[bold]Summary:[/bold] {summary}")


def _display_journal(journal: EvolutionJournal) -> None:
    """Pretty-print the evolution journal as a tree."""
    tree = Tree("[bold]Evolution Journal[/bold]")

    current_phase = ""
    phase_branch = tree

    for event in journal.events:
        # Group by phase
        if event.phase and event.phase != current_phase:
            current_phase = event.phase
            phase_branch = tree.add(f"[bold blue]{current_phase}[/bold blue]")

        label = f"[dim]{event.timestamp[:19]}[/dim] "
        etype = event.event_type

        if etype == EventType.STATE_TRANSITION:
            label += (
                f"[cyan]TRANSITION[/cyan] "
                f"{event.data.get('from', '?')} → {event.data.get('to', '?')}"
            )
        elif etype == EventType.DECISION:
            label += (
                f"[green]DECISION[/green] {event.data.get('decision', '')}\n"
                f"  [dim]Reasoning: {event.data.get('reasoning', '')[:]}"
            )
        elif etype == EventType.LLM_CALL:
            label += (
                f"[magenta]LLM_CALL[/magenta] "
                f"model={event.data.get('model', '?')} "
                f"hash={event.data.get('prompt_hash', '?')[:8]}"
            )
        elif etype == EventType.METRIC_MEASUREMENT:
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in event.data.items())
            label += f"[yellow]METRIC[/yellow] {metrics_str}"
        elif etype == EventType.GUARD_FAILURE:
            label += (
                f"[red]GUARD_FAILURE[/red] "
                f"{event.data.get('guard', '')}; {event.data.get('reason', '')}"
            )
        elif etype == EventType.ROLLBACK_REASON:
            label += (
                f"[red]ROLLBACK[/red] {event.data.get('reason', '')}\n"
                f"  Adjustments: {event.data.get('adjustments', [])}"
            )
        elif etype == EventType.CAPABILITY_ADDED:
            label += (
                f"[blue]CAPABILITY[/blue] +{event.data.get('capability', '')} "
                f"; {event.data.get('reason', '')}"
            )
        elif etype == EventType.PHASE_RESULT:
            result_str = json.dumps(event.data, indent=2, default=str)[:200]
            label += f"[green]RESULT[/green]\n{result_str}"
        else:
            label += f"[dim]{etype.value}[/dim] {json.dumps(event.data, default=str)[:100]}"

        if event.phase:
            phase_branch.add(label)
        else:
            tree.add(label)

    console.print(tree)
    console.print(f"\n[dim]Total events: {len(journal)}[/dim]")
    llm_calls = len(journal.get_events_by_type(EventType.LLM_CALL))
    console.print(f"[dim]LLM calls: {llm_calls} · total tokens: {journal.total_tokens}[/dim]")


# Support for `from stem_agent.cli import app` in typer
if __name__ == "__main__":
    app()
