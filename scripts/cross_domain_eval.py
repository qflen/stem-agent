"""Cross-domain 2×2 evaluation; does each specialization travel?

Reads the seed-0 journals for both domains, extracts the specialized
prompt each one ended up with, then runs that prompt against the
*other* domain's validation slice. Saves the resulting four cells
(CQ-prompt × CQ-corpus, CQ-prompt × sec-corpus, etc.) next to the
seed artifacts so the writeup can name a single source of truth.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from stem_agent.adapters.openai_adapter import OpenAIAdapter
from stem_agent.capabilities.dispatcher import maybe_make_dispatcher
from stem_agent.capabilities.registry import build_default_registry
from stem_agent.core.config import StemAgentConfig
from stem_agent.evaluation.benchmark import make_llm_review_fn, run_benchmark
from stem_agent.evaluation.fixtures.code_samples import (
    BenchmarkSample,
    get_benchmark_corpus,
    partition,
)
from stem_agent.evaluation.fixtures.security_audit_samples import (
    get_security_audit_corpus,
)


@dataclass(frozen=True)
class Cell:
    prompt_domain: str
    corpus_domain: str
    precision: float
    recall: float
    f1: float
    specificity: float


def _seed0_specialized_prompt(seeds_dir: Path) -> str:
    journal_path = seeds_dir / "journal_seed0.json"
    blob = json.loads(journal_path.read_text())
    for event in blob.get("events", []):
        if event.get("event_type") != "phase_result":
            continue
        if event.get("phase") != "specialization":
            continue
        # The phase result records prompt_length; the actual prompt body
        # is stored under the run's ``prompts/`` directory keyed by hash.
        # For the cross-domain run we re-derive the prompt by reading the
        # latest LLM_CALL phase=validation_specialized event's hash and
        # looking it up in the prompts archive.
        break
    # Fall back: read the last validation_specialized LLM_CALL prompt hash.
    last_hash: str | None = None
    for event in blob.get("events", []):
        if event.get("event_type") != "llm_call":
            continue
        if event.get("phase") != "validation_specialized":
            continue
        last_hash = event.get("data", {}).get("prompt_hash")
    if last_hash is None:
        raise RuntimeError(f"no validation_specialized LLM_CALL hash in {journal_path}")
    prompt_path = seeds_dir / "prompts" / f"{last_hash}.txt"
    if not prompt_path.exists():
        raise RuntimeError(
            f"prompt body missing; re-run with --store-prompts: {prompt_path}"
        )
    return prompt_path.read_text()


def _validation_corpus(domain: str, seed: int) -> list[BenchmarkSample]:
    base = get_benchmark_corpus() if domain == "code_quality_analysis" else get_security_audit_corpus()
    return list(partition(base, seed=seed).validation)


def _evaluate_cell(prompt: str, corpus: list[BenchmarkSample], adapter: OpenAIAdapter) -> Cell:
    review_fn = make_llm_review_fn(
        adapter,
        model=StemAgentConfig().execution_model,
        use_tools=True,
        dispatcher=maybe_make_dispatcher(build_default_registry()),
    )
    _, metrics = run_benchmark(review_fn, prompt, corpus=corpus)
    return Cell(
        prompt_domain="",
        corpus_domain="",
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        specificity=metrics.specificity,
    )


def main() -> int:
    cq_dir = Path("docs/example_run/seeds/cq")
    sec_dir = Path("docs/example_run/seeds/sec")
    if not cq_dir.exists() or not sec_dir.exists():
        print("seed directories missing; run differentiate --seeds 3 first")
        return 1

    config = StemAgentConfig()
    if not config.openai_api_key:
        print("OPENAI_API_KEY required for cross-domain evaluation")
        return 1
    adapter = OpenAIAdapter(config)

    cq_prompt = _seed0_specialized_prompt(cq_dir)
    sec_prompt = _seed0_specialized_prompt(sec_dir)

    cells: list[Cell] = []
    for prompt_label, prompt in (("code_quality_analysis", cq_prompt), ("security_audit", sec_prompt)):
        for corpus_label in ("code_quality_analysis", "security_audit"):
            corpus = _validation_corpus(corpus_label, seed=0)
            cell = _evaluate_cell(prompt, corpus, adapter)
            cells.append(
                Cell(
                    prompt_domain=prompt_label,
                    corpus_domain=corpus_label,
                    precision=cell.precision,
                    recall=cell.recall,
                    f1=cell.f1,
                    specificity=cell.specificity,
                )
            )

    output = Path("docs/example_run/seeds/cross_domain.json")
    output.write_text(json.dumps([asdict(c) for c in cells], indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
