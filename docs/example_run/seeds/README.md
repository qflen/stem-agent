# Multi-seed live runs

Three seeds × two domains × `gpt-4o-mini` for sensing/planning/cap-gen
and `gpt-4o` for review execution. Every prompt the LLM saw is archived
under `cq/prompts/` and `sec/prompts/` keyed by its 16-character SHA-256
hash; replay any of them with `stem-agent replay <hash> --prompts-dir
docs/example_run/seeds/{cq,sec}/prompts`.

## Layout

```
seeds/
├── cq/                        # code_quality_analysis, 19→11/4/4 partition
│   ├── journal_seed0.json
│   ├── journal_seed1.json
│   ├── journal_seed2.json
│   └── prompts/<hash>.txt
├── sec/                       # security_audit, 8→4/2/2 (overlapping)
│   ├── journal_seed0.json
│   ├── journal_seed1.json
│   ├── journal_seed2.json
│   └── prompts/<hash>.txt
└── cross_domain.json          # 2×2 prompt × corpus grid
```

## Headline (mean ± 95% CI, percentile bootstrap, 1000 resamples)

### Code-quality (11-sample validation slice)

| metric      |  mean | 95% CI            |
|-------------|------:|-------------------|
| precision   | 0.778 | [0.667, 0.889]    |
| recall      | 0.921 | [0.880, 1.000]    |
| F1          | 0.837 | [0.800, 0.889]    |
| specificity | 0.444 | [0.333, 0.500]    |

### Security audit (4-sample validation slice)

| metric      |  mean | 95% CI            |
|-------------|------:|-------------------|
| precision   | 0.667 | [0.500, 0.750]    |
| recall      | 1.000 | [1.000, 1.000]    |
| F1          | 0.794 | [0.667, 0.857]    |
| specificity | 0.000 | [0.000, 0.000]    |

Regenerate either table with `stem-agent report docs/example_run/seeds/{cq,sec}`.

## Per-seed outcomes

| Domain | Seed | Final | Rollbacks | Spec F1 | Generated capability                  | Holdout |
|--------|------|-------|-----------|---------|----------------------------------------|---------|
| cq     | 0    | spec. | 0         | 0.800   | `contextual_analysis`                  | rejected (1 vs 1) |
| cq     | 1    | spec. | 0         | 0.889   | `contextual_analysis`                  | rejected (0 vs 0) |
| cq     | 2    | spec. | 0         | 0.824   | `dependency_analysis`                  | rejected (2 vs 2) |
| sec    | 0    | spec. | 2         | 0.857   | `subprocess_context_analysis`          | bypassed (overlapping) |
| sec    | 1    | spec. | 0         | 0.667   | `contextual_vulnerability_analysis`    | bypassed (overlapping) |
| sec    | 2    | spec. | 0         | 0.857   | `contextual_vulnerability_analysis`    | bypassed (overlapping) |

Notes:

- Every CQ proposal cleared the AST scan + sandbox but the empirical
  holdout gate rejected each one (the with-proposal arm did not strictly
  beat the without-proposal arm). The capabilities are therefore *not*
  in the specialized prompt for any CQ run.
- Security holdouts are bypassed by design: the 8-sample security
  corpus partitions to 4 / 2 / 2 with `overlapping=True`, and the gate
  refuses to admit on too-small slices. The proposal gets in
  post-sandbox, which is the writeup's "demonstration, not load-bearing"
  caveat.
- Sec seed 0 is the rollback proof point on this batch: 2 attempts
  before the guard cleared, with cross-check disagreements feeding
  diagnose_failure on each.

## Cross-domain 2×2

`scripts/cross_domain_eval.py` takes the seed-0 specialized prompt
from each domain and runs it against the *other* domain's validation
slice. Output: `cross_domain.json`.

| Prompt                  | Corpus                  |  P    |   R   |   F1  |  Spec |
|-------------------------|-------------------------|------:|------:|------:|------:|
| code\_quality\_analysis | code\_quality\_analysis | 0.500 | 1.000 | 0.667 | 0.111 |
| code\_quality\_analysis | security\_audit         | 0.429 | 1.000 | 0.600 | 0.000 |
| security\_audit         | code\_quality\_analysis | 0.500 | 0.875 | 0.636 | 0.000 |
| security\_audit         | security\_audit         | 0.750 | 1.000 | 0.857 | 0.000 |

Off-diagonals do **not** beat on-diagonals; the security specialization
is brittle when transferred (drops 0.22 F1) while the code-quality
specialization is roughly flat across corpora.

## Cohen's κ (label-only `gpt-4o-mini` second rater)

`scripts/adjudicate.py` relabels each corpus with a strict
single-token-only prompt; the per-sample labels live next to the
fixtures.

| Corpus                  | κ    | 95% CI         | Disputed |
|-------------------------|-----:|---------------:|---------:|
| code\_quality\_analysis | 0.546 | [0.307, 0.785] | 7 / 19  |
| security\_audit         | 0.000 | [0.000, 1.000] | 2 /  8  |

The security κ collapse is informative: the rater labels every
security sample "security" (including the two clean adversarial
ones), so observed agreement (75%) exactly matches chance agreement
and κ → 0. A one-shot LLM rater without context is not trustworthy
ground truth for adversarial clean code.

## Total live spend

`cumulative_spend()` walks the per-seed journals; cross-domain and
adjudicator costs aren't journaled but are estimated from per-call
token counts at the published OpenAI rates.

| Item                | Tokens     | Cost   |
|---------------------|-----------:|-------:|
| 3 cq seeds          | 61,063     | $0.184 |
| 3 sec seeds         | 36,755     | $0.092 |
| Cross-domain 2×2    | ~162k est  | ~$0.89 |
| Cohen's κ rater     | ~5k est    | ~$0.002 |
| **Total**           | **~265k**  | **~$1.17** |

The CLI's `--seeds` loop computes `cumulative_spend(out_dir)` before
each iteration and aborts at the configured cap (default $25).

## Reproducing

```bash
export OPENAI_API_KEY=sk-…
.venv/bin/stem-agent differentiate --seeds 3 --domain code_quality_analysis --store-prompts
.venv/bin/stem-agent differentiate --seeds 3 --domain security_audit       --store-prompts
.venv/bin/python scripts/cross_domain_eval.py
.venv/bin/python scripts/adjudicate.py
.venv/bin/stem-agent report docs/example_run/seeds/cq
.venv/bin/stem-agent report docs/example_run/seeds/sec
```

OpenAI honours `seed` on chat completions as best-effort, not a hard
reproducibility guarantee; re-running may produce slightly different
metric values within the bootstrap CI bounds reported above.
