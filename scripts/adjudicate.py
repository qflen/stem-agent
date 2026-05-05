"""Cohen's κ second-rater pass; relabel both corpora with gpt-4o-mini.

Calls the model with a strict label-only prompt (no chain-of-thought) so
the answer space is constrained to the same canonical categories the
benchmark uses. Computes κ vs. the original corpus labels with a
bootstrap CI and saves the adjudicator's labels next to the corpus
fixtures so the writeup can be re-rendered without re-running the
adjudicator every time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stem_agent.adapters.openai_adapter import OpenAIAdapter
from stem_agent.core.config import StemAgentConfig
from stem_agent.evaluation.benchmark import normalize_category
from stem_agent.evaluation.fixtures.code_samples import (
    BenchmarkSample,
    get_benchmark_corpus,
)
from stem_agent.evaluation.fixtures.security_audit_samples import (
    get_security_audit_corpus,
)
from stem_agent.evaluation.kappa import disputed_count, kappa_with_ci

_LABEL_PROMPT = (
    "You are labelling a Python code snippet for the kind of issue it contains. "
    "Reply with EXACTLY ONE of: clean, logic, security, structure, performance. "
    "No explanation. No JSON. Just the single word."
)


def _truth_label(sample: BenchmarkSample) -> str:
    if sample.is_clean or not sample.issue_categories:
        return "clean"
    return normalize_category(sample.issue_categories[0])


def _adjudicate_one(adapter: OpenAIAdapter, sample: BenchmarkSample) -> str:
    prompt = (
        f"{_LABEL_PROMPT}\n\n## Code\n```python\n{sample.code}\n```\nLabel:"
    )
    raw = adapter.generate(prompt, model="gpt-4o-mini").strip().lower().splitlines()[0]
    cleaned = raw.strip(" ,.!?:;\"'`*")
    if cleaned not in {"clean", "logic", "security", "structure", "performance"}:
        return "clean"
    return cleaned


def _adjudicate_corpus(name: str, samples: list[BenchmarkSample], adapter: OpenAIAdapter) -> dict:
    truth = [_truth_label(s) for s in samples]
    pred = [_adjudicate_one(adapter, s) for s in samples]
    result = kappa_with_ci(truth, pred, n_resamples=1000)
    out = {
        "domain": name,
        "kappa": result.kappa,
        "ci_lo": result.lo,
        "ci_hi": result.hi,
        "disputed": disputed_count(truth, pred),
        "labels": {s.sample_id: pred[i] for i, s in enumerate(samples)},
    }
    return out


def main() -> int:
    config = StemAgentConfig()
    if not config.openai_api_key:
        print("OPENAI_API_KEY required for adjudication")
        return 1
    adapter = OpenAIAdapter(config)
    fixtures_dir = Path("src/stem_agent/evaluation/fixtures")

    cq_result = _adjudicate_corpus("code_quality_analysis", get_benchmark_corpus(), adapter)
    sec_result = _adjudicate_corpus("security_audit", get_security_audit_corpus(), adapter)

    (fixtures_dir / "adjudicator_labels_cq.json").write_text(json.dumps(cq_result, indent=2))
    (fixtures_dir / "adjudicator_labels_sec.json").write_text(json.dumps(sec_result, indent=2))

    for result in (cq_result, sec_result):
        print(
            f"{result['domain']}: κ = {result['kappa']:.3f} "
            f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}] "
            f"disputed={result['disputed']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
