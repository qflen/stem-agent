# Example differentiation run

A real run against the OpenAI API, captured so reviewers do not have to
spend tokens to see the pipeline work end-to-end. The full journal is
checked in next to this file as `journal.json`.

## Run metadata

| Field | Value |
|---|---|
| Started | 2026-04-17T09:21:55Z |
| Finished | 2026-04-17T09:26:46Z |
| Wall clock | ~4m 51s |
| Domain | `code_quality_analysis` |
| Planning model | `gpt-4o-mini` |
| Execution model | `gpt-4o` |
| LLM calls | 42 |
| Total tokens | 40,909 |
| State transitions | undifferentiated → sensing → differentiating → validating → specialized |
| Final state | `SPECIALIZED` (no rollbacks) |

## Token usage by phase

| Phase | Calls | Tokens |
|---|---:|---:|
| sensing | 1 | 1,218 |
| planning | 1 | 1,546 |
| validation_baseline (20-sample benchmark) | 20 | 17,248 |
| validation_specialized (20-sample benchmark) | 20 | 20,897 |

## Benchmark outcome

| Metric | Baseline (undifferentiated) | Specialized | Δ |
|---|---:|---:|---:|
| Precision | 0.000 | 0.667 | +0.667 |
| Recall | 0.000 | 0.933 | +0.933 |
| F1 | 0.000 | 0.778 | +0.778 |
| Specificity | 0.000 | 0.300 | +0.300 |

The baseline score is zero because the undifferentiated prompt does not
ask for structured JSON — the parser falls back to an empty category
set for every sample, so precision and recall are both undefined and
report as 0. That is the signal the differentiation pipeline is there
to fix, and the delta above is what it actually buys you.

## What the cross-check caught

After the specialized benchmark run, the validation phase replays each
verdict through the deterministic static checks and logs any
disagreements as DECISION events. On this run it logged three:

- `smell_03` and `clean_03` — the LLM flagged structural issues the
  AST metrics did not back up (short functions, shallow nesting).
- `clean_03` — the pattern scanner spotted a bare `eval()` call the
  LLM missed.

These are the kinds of discrepancies an agent monitoring itself should
surface rather than paper over. They live in the journal alongside
everything else so a reviewer can replay the reasoning.

## How to reproduce

```
.venv/bin/stem-agent differentiate --domain code_quality_analysis
```

Rough cost at current OpenAI pricing is about twenty cents per run.
